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
    gumbel_argmax_from_logits: bool, optional (default = False)
        If Gumbel is and we decide to do argmax at inference, this allows to use argmax over logits and not over Gumbel
        output
    infer_with_argmax: bool, optional (default = True)
        This allows to multiply target embedding matrix by argmax of output of (Gumbel) softmax at inference time
    detach_self_feeding: bool, optional (default = True)
        During self-feeding (scheduled sampling ratio > 0) we pass the produced word embedding to the next decoder
        timestep. However, it is now differentiable, so this parameter allows to break computational graph to make the
        whole thing match the case of vanilla seq2seq.

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
                 gumbel_tau: float = 1.0,
                 gumbel_hard: bool = True,
                 gumbel_eps: float = 1e-10,
                 gumbel_argmax_from_logits: bool = False,
                 infer_with_argmax: bool = True,
                 detach_self_feeding: bool = True,
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
        self._infer_with_argmax = infer_with_argmax
        self._detach_self_feeding = detach_self_feeding
        self._gumbel_argmax_from_logits = gumbel_argmax_from_logits

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
        step_probabilities = []
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
                embedded_decoder_input = self._target_embedder(input_choices)  # teacher forcing input
            else:
                if timestep == 0:
                    # For the first timestep, when we do not have targets, we input start symbols.
                    # (batch_size,)
                    input_choices = source_mask.new_full((batch_size,), fill_value=self._start_index)
                    embedded_decoder_input = self._target_embedder(input_choices)
                else:
                    embedded_decoder_input = embedded_output  # at inference time feed softmax*embedding_matrix vectors

            # input_choices : (batch_size,)  since we are processing these one timestep at a time.
            # (batch_size, target_embedding_dim)

            decoder_input = self._prepare_decode_step_input(embedded_decoder_input, decoder_hidden,
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
                                                       tau=self.gumbel_tau, hard=self.gumbel_hard, eps=self.gamble_eps)
            else:
                raise ValueError("Wrong calculation fucntion. Should be either 'gumbel' or 'softmax'.")

            step_probabilities.append(class_probabilities.unsqueeze(1))

            if self._weights_calculation_function == 'gumbel' and self._gumbel_argmax_from_logits:
                argmax_classes = torch.argmax(output_projections, 1)
            else:
                argmax_classes = torch.argmax(class_probabilities, 1)
            last_argmax_classes = argmax_classes

            if self.training:  # if training, we produce differentiable vector based on (Gumbel) softmax
                if self._detach_self_feeding:
                    # if we want to break computational graph at self-feeding point, just how it is in usual NMT
                    embedded_output = torch.matmul(class_probabilities.detach(), self._target_embedder.weight)
                else:  # it is possible due to differentiability of `class_probabilities` and not possible in usual NMT
                    embedded_output = torch.matmul(class_probabilities, self._target_embedder.weight)
            else:  # if inference
                if self._infer_with_argmax:  # use argmax sampling
                    embedded_output = self._target_embedder(last_argmax_classes)
                else:
                    embedded_output = torch.matmul(class_probabilities, self._target_embedder.weight)

            # (batch_size, 1)
            step_argmax_classes.append(last_argmax_classes.unsqueeze(1))
        # step_logits is a list containing tensors of shape (batch_size, 1, num_classes)
        # This is (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)
        class_probabilities = torch.cat(step_probabilities, 1)
        all_argmax_classes = torch.cat(step_argmax_classes, 1)
        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,
                       "predictions": all_argmax_classes}
        if tgt_tokens:
            target_mask = util.get_text_field_mask(tgt_tokens)
            loss = self._get_loss(logits, targets, target_mask)
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
    def _get_loss(logits: torch.LongTensor,
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
        loss = util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
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
