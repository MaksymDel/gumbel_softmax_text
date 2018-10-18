from typing import Dict

import numpy
import torch
import torch.nn.functional as F
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.attention import LegacyAttention
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from overrides import overrides
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

# TODO: FIX LOSS COMPUTATION WITH GUMBEL ONE HOT!

@Model.register("differentiable_nll")
class Rnn2RnnDifferentiableNll(Model):
    """
    Vanilla seq2seq NMT model that produces differentiable output (which is useful e.g. in GANs).

    Consider following perspective. The output of NMT system is a list of one-hot vectors representing output words.
    Having this list, we can either obtain output word embeddings by matmul with embedding matrix or index target
    vocabulary and obtain word strings. In the first case, however, the result vectors are non differentiable,
    because we obtained one-hot vectors using argmax over logits (derivative of argmax is zero).

    There are two workarounds that model offers:
    1) using plain softmax distribution instead of one-hot vectors.
    2) obtaining one-hot vectors via Gumbel softamax

    In the first case, we simply multiply embedding matrix by softmax weights, and get a point in target embedding
    space (no argmax involved -> differentiable). In the second case, our one-hot vectors are differentiable since
    the are the result of differentiable Gumbel softmax operation. It might be better, because it might prevent
    potential GAN discriminator from learning to separate one-hot based embeddings versus softmax based embeddings.

    Uses cross-entropy loss as usual seq2seq models.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (``tokens``) or the target tokens can have a different namespace, in which case it needs to
        be specified as ``target_namespace``.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : int, required
        Length of decoded sequences
    target_namespace : str, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : int, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention_function: ``SimilarityFunction``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    scheduled_sampling_ratio: float, optional (default = 0.0)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015.
    weight_function, optional (options = {"gumbel", "softmax"}, default = "softmax")
        Either to multiply target embedding matrix by softmax distribution or by gumbel output
    gumbel_tau: float, optional (default = 1.0)
        Temperature in Gumbel softmax
    gumbel_hard: bool, optional (default = True)
        Gumbel softmax can also return soft distribution (similar to softmax) if this param equals False
    gumbel_eps: float, optional (default = 1e-10)
        Epsilon parameter in Gumbel softmax
    self_feed_with: string, optional (default = distribution
        During self feeding we can safely do argmax:
        Options:
        1) "distribution" = multiply embeddings matrix by output of (Gumbel) softmax. Keeps embedding differentiable
        2) "argmax_logits" = take argmax over logits (before applying Gumbel) and use result onehot vectors
        to get word embedding. Makes embedding non differentiable (but it is just self feeding - input to the next
        decoder timestep, output embeddings are still diffrenetiable)
        3) "argmax_distribution" = the same as "argmax_logits", but argmax is applied to the output of (Gumbel) softmax
        4) "detach_distribution" - the same as "distribution", but allows to break computational graph to make the
        whole thing match the case of vanilla seq2seq.
    infer_with: string, optional (default = distribution)
        During inference we can optionally do argmax (will make output non-differentiable however).
        It these 3 options are the same in case of softmax
        Options:
        1) "distribution" - multiply embeddings matrix by output of Gumbel. Keeps differentiable
        Output (strings) is  the same as argmax_distribution. The same as "argmax_distribution" in this experiment
        2) "argmax_logits" - take argmax over logits (before applying Gumbel) and use result onehot vectors
        to get word embedding.
        3) "argmax_distribution" - the same as "argmax_logits", but argmax is applied to the output of (Gumbel) softmax

    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 attention_function: SimilarityFunction = None,
                 scheduled_sampling_ratio: float = 0.0,
                 weight_function="softmax",
                 gumbel_tau: float = 0.66,
                 gumbel_hard: bool = True,
                 gumbel_eps: float = 1e-10,
                 infer_with: str = "distribution",
                 self_feed_with: str = "argmax_distribution",
                 ) -> None:
        super(Rnn2RnnDifferentiableNll, self).__init__(vocab)
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._attention_function = attention_function
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder. Also, if
        # we're using attention with ``DotProductSimilarity``, this is needed.
        self._decoder_output_dim = self._encoder.get_output_dim()
        target_embedding_dim = target_embedding_dim or self._source_embedder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)
        if self._attention_function:
            self._decoder_attention = LegacyAttention(self._attention_function)
            # The output of attention, a weighted average over encoder outputs, will be
            # concatenated to the input vector of the decoder at each time step.
            self._decoder_input_dim = self._encoder.get_output_dim() + target_embedding_dim
        else:
            self._decoder_input_dim = target_embedding_dim
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)

        self._weights_calculation_function = weight_function

        self._gumbel_tau = gumbel_tau
        self._gumbel_hard = gumbel_hard
        self._gamble_eps = gumbel_eps

        if self_feed_with not in {"distribution", "argmax_logits", "argmax_distribution", "detach_distribution"}:
            raise ValueError(
                "Allowed vals for selffeed are {distribution, argmax_logits, argmax_distribution, detach_distribution}")

        if infer_with not in {"distribution", "argmax_logits", "argmax_distribution"}:
            raise ValueError(
                "Allowed vals for ifer_with are {distribution, argmax_logits, argmax_distribution}")

        self._infer_with = infer_with
        self._self_feed_with = self_feed_with



    @overrides
    def forward(self,  # type: ignore
                src_tokens: Dict[str, torch.LongTensor],
                tgt_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.

        Parameters
        ----------
        src_tokens : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        tgt_tokens : Dict[str, torch.LongTensor], optional (default = None)
           Output of ``Textfield.as_array()`` applied on target ``TextField``. We assume that the
           target tokens are also represented as a ``TextField``.
        """
        # (batch_size, input_sequence_length, encoder_output_dim)
        embedded_src_tokens = self._source_embedder(src_tokens)
        batch_size, _, _ = embedded_src_tokens.size()
        source_mask = util.get_text_field_mask(src_tokens)
        encoder_outputs = self._encoder(embedded_src_tokens, source_mask)
        # (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            encoder_outputs,
            source_mask,
            self._encoder.is_bidirectional()
        )
        if tgt_tokens:
            targets = tgt_tokens["tokens"]
            target_sequence_length = targets.size()[1]
            # The last input from the target is either padding or the end symbol. Either way, we
            # don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps
        decoder_hidden = final_encoder_output
        decoder_context = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
        step_logits = []
        step_predicted_embeddings = []
        step_probabilities = []
        if not self.training:
            step_argmax_classes = []

        for timestep in range(num_decoding_steps):
            use_gold_targets = False
            # Use gold tokens at test time when provided and at a rate of 1 -
            # _scheduled_sampling_ratio during training.
            if self.training:
                if torch.rand(1).item() >= self._scheduled_sampling_ratio:
                    use_gold_targets = True
            elif tgt_tokens:
                use_gold_targets = True

            if use_gold_targets:
                input_choices = targets[:, timestep]
                embedded_next_token_to_decoder = self._target_embedder(input_choices)  # teacher forcing input
            else:
                if timestep == 0:
                    # For the first timestep, when we do not have targets, we input start symbols.
                    # (batch_size,)
                    input_choices = source_mask.new_full((batch_size,), fill_value=self._start_index)
                    embedded_next_token_to_decoder = self._target_embedder(input_choices)
                else:  # at inference time feed softmax*embedding_matrix vectors
                    embedded_next_token_to_decoder = embedded_token_to_self_feed

            # input_choices : (batch_size,)  since we are processing these one timestep at a time.
            # (batch_size, target_embedding_dim)

            decoder_input = self._prepare_decode_step_input(embedded_next_token_to_decoder, decoder_hidden,
                                                            encoder_outputs, source_mask)

            decoder_hidden, decoder_context = self._decoder_cell(decoder_input,
                                                                 (decoder_hidden, decoder_context))
            # (batch_size, num_classes)
            output_projections = self._output_projection_layer(decoder_hidden)
            # list of (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            if self._weights_calculation_function == 'softmax':
                class_probabilities = F.softmax(output_projections, dim=-1)
            elif self._weights_calculation_function == 'gumbel':
                class_probabilities = F.gumbel_softmax(output_projections,
                                                       tau=self._gumbel_tau, hard=self._gumbel_hard, eps=self._gamble_eps)
            else:
                raise ValueError("Wrong calculation fucntion. Should be either 'gumbel' or 'softmax'.")

            step_probabilities.append(class_probabilities.unsqueeze(1))

            # assign which token to self-feed (pass to the next decoder timestep)
            embedded_token_to_self_feed = None
            if self._self_feed_with == "distribution":
                embedded_token_to_self_feed = torch.matmul(class_probabilities, self._target_embedder.weight)
            elif self._self_feed_with == "detach_distribution":
                embedded_token_to_self_feed = torch.matmul(class_probabilities.detach(), self._target_embedder.weight)
            elif self._self_feed_with == "argmax_distribution":
                embedded_token_to_self_feed = self._target_embedder(torch.argmax(class_probabilities, 1))
            elif self._self_feed_with == "argmax_logits":
                embedded_token_to_self_feed = self._target_embedder(torch.argmax(output_projections, 1))

            # assign which token to
            embedded_token_to_return = None
            if self.training:  # during training we always return differentiable tokens
                if self._self_feed_with == "distribution":  # return differentiable token to self-feed
                    embedded_token_to_return = embedded_token_to_self_feed
                else:  # or make a new one, which is again differentiable
                    embedded_token_to_return = torch.matmul(class_probabilities, self._target_embedder.weight)
                assert embedded_token_to_return.requires_grad
            else:  # at inference we might return non-differentiable token as well
                if self._infer_with == "distribution":  # output is same as argmax_distribution
                    last_argmax_classes = torch.argmax(class_probabilities, 1)
                    embedded_token_to_return = torch.matmul(class_probabilities, self._target_embedder.weight)
                elif self._infer_with == "argmax_distribution":
                    last_argmax_classes = torch.argmax(class_probabilities, 1)
                    embedded_token_to_return = self._target_embedder(last_argmax_classes)
                elif self._infer_with == "argmax_logits":
                    last_argmax_classes = torch.argmax(output_projections, 1)
                    embedded_token_to_return = self._target_embedder(last_argmax_classes)

            step_predicted_embeddings.append(embedded_token_to_return.unsqueeze(1))

            if not self.training:
                # (batch_size, 1)
                step_argmax_classes.append(last_argmax_classes.unsqueeze(1))
        # step_logits is a list containing tensors of shape (batch_size, 1, num_classes)
        # This is (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)
        class_probabilities = torch.cat(step_probabilities, 1)
        result_embeddings = torch.cat(step_predicted_embeddings, 1)

        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,
                       "result_embeddings": result_embeddings}

        if not self.training:
            all_argmax_classes = torch.cat(step_argmax_classes, 1)
            output_dict["predictions"] = all_argmax_classes

        if tgt_tokens:
            target_mask = util.get_text_field_mask(tgt_tokens)
            loss = self._get_loss(class_probabilities, targets, target_mask)
            output_dict["loss"] = loss
            # TODO: Define metrics
        return output_dict

    def _prepare_decode_step_input(self,
                                   embedded_input: torch.LongTensor,
                                   decoder_hidden_state: torch.LongTensor = None,
                                   encoder_outputs: torch.LongTensor = None,
                                   encoder_outputs_mask: torch.LongTensor = None) -> torch.LongTensor:
        """
        Given the input indices for the current timestep of the decoder, and all the encoder
        outputs, compute the input at the current timestep.  Note: This method is agnostic to
        whether the indices are gold indices or the predictions made by the decoder at the last
        timestep. So, this can be used even if we're doing some kind of scheduled sampling.

        If we're not using attention, the output of this method is just an embedding of the input
        indices.  If we are, the output will be a concatentation of the embedding and an attended
        average of the encoder inputs.

        Parameters
        ----------
        input_indices : torch.LongTensor
            Indices of either the gold inputs to the decoder or the predicted labels from the
            previous timestep.
        decoder_hidden_state : torch.LongTensor, optional (not needed if no attention)
            Output of from the decoder at the last time step. Needed only if using attention.
        encoder_outputs : torch.LongTensor, optional (not needed if no attention)
            Encoder outputs from all time steps. Needed only if using attention.
        encoder_outputs_mask : torch.LongTensor, optional (not needed if no attention)
            Masks on encoder outputs. Needed only if using attention.
        """
        if self._attention_function:
            # encoder_outputs : (batch_size, input_sequence_length, encoder_output_dim)
            # Ensuring mask is also a FloatTensor. Or else the multiplication within attention will
            # complain.
            encoder_outputs_mask = encoder_outputs_mask.float()
            # (batch_size, input_sequence_length)
            input_weights = self._decoder_attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)
            # (batch_size, encoder_output_dim)
            attended_input = util.weighted_sum(encoder_outputs, input_weights)
            # (batch_size, encoder_output_dim + target_embedding_dim)
            return torch.cat((attended_input, embedded_input), -1)
        else:
            return embedded_input

    @staticmethod
    def _get_loss(class_probs: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.LongTensor:
        """
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        relevant_targets = targets[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        loss = Rnn2RnnDifferentiableNll._sequence_cross_entropy_with_probs(class_probs, relevant_targets, relevant_mask)
        return loss

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    @staticmethod
    def _sequence_cross_entropy_with_probs(probs: torch.FloatTensor,
                                           targets: torch.LongTensor,
                                           weights: torch.FloatTensor,
                                           batch_average: bool = True,
                                           label_smoothing: float = None) -> torch.FloatTensor:
        """
        Computes the cross entropy loss of a sequence, weighted with respect to
        some user provided weights. Note that the weighting here is not the same as
        in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
        classes; here we are weighting the loss contribution from particular elements
        in the sequence. This allows loss computations for models which use padding.

        Parameters
        ----------
        probs : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
            which contains the normalized probability for each class.
        targets : ``torch.LongTensor``, required.
            A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
            index of the true class for each corresponding step.
        weights : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of size (batch, sequence_length)
        batch_average : bool, optional, (default = True).
            A bool indicating whether the loss should be averaged across the batch,
            or returned as a vector of losses per batch element.
        label_smoothing : ``float``, optional (default = None)
            Whether or not to apply label smoothing to the cross-entropy loss.
            For example, with a label smoothing value of 0.2, a 4 class classifcation
            target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
            the correct label.

        Returns
        -------
        A torch.FloatTensor representing the cross entropy loss.
        If ``batch_average == True``, the returned loss is a scalar.
        If ``batch_average == False``, the returned loss is a vector of shape (batch_size,).

        """
        # shape : (batch * sequence_length, num_classes)
        probs_flat = probs.view(-1, probs.size(-1))

        # sometimes probs are onehot vector, so to take log later we should smooth

        probs_flat += 1e-35

        # shape : (batch * sequence_length, num_classes)
        log_probs_flat = probs_flat.log()
        # shape : (batch * max_len, 1)
        targets_flat = targets.view(-1, 1).long()

        if label_smoothing is not None and label_smoothing > 0.0:
            num_classes = probs.size(-1)
            smoothing_value = label_smoothing / num_classes
            # Fill all the correct indices with 1 - smoothing value.
            one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
            smoothed_targets = one_hot_targets + smoothing_value
            negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
            negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
        else:
            # Contribution to the negative log likelihood only comes from the exact indices
            # of the targets, as the target distributions are one-hot. Here we use torch.gather
            # to extract the indices of the num_classes dimension which contribute to the loss.
            # shape : (batch * sequence_length, 1)
            negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood * weights.float()
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)

        if batch_average:
            num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
            return per_batch_loss.sum() / num_non_empty_sequences

        return per_batch_loss
