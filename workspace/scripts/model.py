import numpy as np
import tensorflow as tf
from scripts.transformer_tutorial_module import positional_encoding, create_padding_mask, create_look_ahead_mask, EncoderLayer

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3


class Generator(tf.keras.Model):
    """
    スレッドタイトルを生成する
    """
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,
                 max_pos_encoding, rate=0.1):
        """
        Args:
          num_layers (int): Transformerのパラメータ、詳細はチュートリアル参照
          d_model (int): Transformerのパラメータ、詳細はチュートリアル参照
          num_heads (int): Transformerのパラメータ、詳細はチュートリアル参照
          dff (int): Transformerのパラメータ、詳細はチュートリアル参照
          vocab_size (int): Embeddingレイヤーの語彙サイズ
          max_pos_encoding (int): 最大シーケンスサイズ
          rate (float): Dropoutレイヤーのドロップ率
        """
        super(Generator, self).__init__()
        self.seq_len = max_pos_encoding

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_pos_encoding, d_model)

        self.stack_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                             for _ in range(num_layers)]

        self.final_layer = tf.keras.layers.Dense(vocab_size)

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        seq_len = x.shape[1]
        padding_mask = create_padding_mask(x)
        look_ahead_mask = create_look_ahead_mask(seq_len)
        combined_mask = tf.maximum(padding_mask, look_ahead_mask)

        # 埋め込みと位置エンコーディングを合算する
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        # (batch_size, input_seq_len, d_model)
        for i in range(self.num_layers):
            x = self.stack_layers[i](x, training=training, mask=combined_mask)

        # (batch_size, input_seq_len, vocab_size)
        x = self.final_layer(x)
        return x

    @classmethod
    def padding(clf, sentences):
        """
        文書をEOSトークンより後ろをすべてPADトークン(0)に置き換える
        例) 入力: [[1, 10, 11, 12, 2, 13, 14, 15]] → 出力: [[1, 10, 11, 12, 2, 0, 0, 0]]

        Args:
          sentences (array): 文書のトークンIDを格納したArray

        Returns:
          array: 生成した文書のトークンIDを格納した配列、サイズは(num_sample, seq_len)
        """
        # sentences.shape == (num_sentences, seq_len)
        ret = []
        for row in sentences:
            try:
                first_eos_idx = row.tolist().index(EOS_TOKEN_ID)
                row[first_eos_idx+1:] = PAD_TOKEN_ID
            except ValueError:
                pass  # EOSトークンがないときはそのまま
            ret.append(row)

        return np.array(ret)

    def sample(self, num_sample, temperature=1.0, prohibit_unk=False, prefix=None, padding=True):
        """
        文書を生成する

        Args:
          num_sample (int): 生成する文書数
          temperature (float): 
          prohibit_unk (bool): UNKトークンを禁止するかどうか
          prefix (Option[list[int]]): 生成する文書の最初のトークンIDを指定する
          padding (bool): EOSトークンを生成した後、Paddingトークン(0)埋めをするかどうか

        Returns:
          array: 生成した文書のトークンIDを格納したArray、サイズは(num_sample, seq_len)
        """
        # (batch_size, 1)
        prefix_ids = [BOS_TOKEN_ID]
        if prefix is not None:
            prefix_ids += prefix
        gen_input = tf.constant(np.tile(prefix_ids, (num_sample, 1)), dtype=tf.int32)

        # 再帰的にcallを呼び出すことで文書を生成する
        for _ in range(self.seq_len - len(prefix_ids)):
            gen_output = self.call(gen_input, False)

            predictions = gen_output[:, -1:, :]  # (batch_size, 1, vocab_size)

            if prohibit_unk:  # UNK TOKEN禁止の場合はUNK_TOKENの出力を小さくする
                predictions[:, -1, UNK_TOKEN_ID] = -999999.9

            if temperature == 0.0:
                predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            else:
                predictions = tf.reshape(predictions, (num_sample, -1))  # (batch_size, vocab_size)
                predictions = predictions / temperature
                predicted_id = tf.random.categorical(predictions, num_samples=1, dtype="int32")

            gen_input = tf.concat([gen_input, predicted_id], axis=-1)

        output = gen_input.numpy()

        if padding:
            output = self.padding(output)

        return output


class Filter(tf.keras.Model):
    """
    入力されたスレッドタイトルの尤もらしさをスコア付するクラス
    """

    def __init__(self, conv_filters, conv_kernel_sizes, d_model, vocab_size, rate=0.5):
        """
        Args:
          conv_filters (list[int]): CNNのfilter数のリスト、conv_kernel_sizesと同じサイズのリストにする
          conv_kernel_sizes (list[int]): CNNのkernel_sizeのリスト、conv_filtersと同じサイズのリストにする
          d_model (int): Embedding, FC層のUnitサイズ
          vocab_size (int): Embeddingレイヤーの語彙サイズ
          rate (float): Dropoutレイヤーのドロップ率
        """
        super(Filter, self).__init__()
        assert len(conv_filters) == len(conv_kernel_sizes)

        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)

        self.conv = []
        for filters, kernel_size in zip(conv_filters, conv_kernel_sizes):
            c = tf.keras.layers.Conv1D(filters, kernel_size, activation="relu", padding="same")
            self.conv.append(c)

        self.dense1 = tf.keras.layers.Dense(d_model, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1)

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        for c in self.conv:
            x = c(x)
            x = tf.keras.layers.MaxPooling1D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x
