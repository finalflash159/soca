---
type: learning_note
domain: deep-learning
topic: attention-and-transformers
status: active
created: 2026-07-22
updated: 2026-07-28
tags: [deep-learning, neural-network, attention, transformer, training]
source_kind: personal-study-simulation
---

# DL: attention và Transformer — từ neuron đến “nhìn phần liên quan”

## Tôi từng tưởng tượng model thế nào

Tôi từng nghĩ neural network là một chuỗi công thức thần bí học cách trả lời.
Cách hiểu dễ dùng hơn với tôi là model biến input thành representation qua nhiều
lớp; mỗi lớp đổi không gian biểu diễn để pattern cần cho task trở nên dễ tách
hơn.

Neuron chỉ là weighted sum rồi đi qua activation. Một layer linear đơn độc chỉ
tạo biến đổi tuyến tính; nhiều layer với activation mới biểu diễn được biên cong
và quan hệ phức tạp.

## Từ loss đến weight

Forward pass tạo prediction. Loss đo prediction lệch target bao nhiêu. Backward
pass dùng chain rule để tính gradient của loss theo từng weight. Optimizer cập
nhật weight theo gradient, learning rate và state như momentum/variance.

Tôi không nói model “hiểu” khi loss giảm. Loss chỉ là objective mà tôi đặt ra;
model có thể học shortcut khiến metric đẹp nhưng hành vi ngoài distribution tệ.

## Attention là phép chọn ngữ cảnh

Với mỗi token, query hỏi “tôi cần thông tin gì?”, key nói “tôi đại diện cho pattern
nào?” và value chứa nội dung được lấy. Attention score so query với key, softmax
thành trọng số rồi trộn value.

```text
Q = XWq
K = XWk
V = XWv
Attention(Q,K,V) = softmax(QKᵀ / √d) V
```

Tôi ví một token đang đọc câu như người làm việc ở bàn: query là câu hỏi trong
đầu, key là nhãn trên các tờ giấy, value là nội dung tờ giấy. Softmax phân bổ
chú ý, không phải một công tắc đúng/sai tuyệt đối.

## Vì sao chia cho √d

Khi chiều vector tăng, dot product có variance lớn. Softmax quá nhọn làm gradient
nhỏ và training khó. Chia cho `√d` giữ score trong vùng dễ học hơn. Đây là scale
ổn định số, không phải mẹo để model “thông minh” hơn.

## Self-attention và cross-attention

Self-attention lấy Q/K/V từ cùng một sequence nên token nhìn các token khác. Causal
self-attention chặn token tương lai khi train decoder, để model không nhìn đáp án
trước.

Cross-attention lấy query từ decoder và key/value từ encoder hoặc nguồn khác. Tôi
liên tưởng đây là lúc một người viết câu trả lời quay lại đọc tài liệu nguồn.

## Multi-head

Một head có thể học quan hệ gần, head khác học subject-verb, vị trí hoặc entity.
Các head không có nhãn “head này chuyên ngữ pháp”; đó là cách tôi diễn giải sau
khi xem pattern. Multi-head rồi concat/project để các góc nhìn quay về dimension
chung.

## Positional information

Attention thuần túy không tự biết thứ tự. Positional encoding hoặc positional
embedding đưa thông tin vị trí vào representation. Với context dài, cách biểu
diễn position ảnh hưởng khả năng extrapolate; không được nói “model có context
32K thì mọi vị trí trong đó đều chất lượng như nhau”.

## Residual và normalization

Residual connection cho layer học delta thay vì phải học lại identity. Nó giúp
gradient và optimization ổn định. Normalization làm scale activation dễ kiểm
soát hơn. Thứ tự pre-norm/post-norm ảnh hưởng training dynamics và implementation.

## Feed-forward block

Attention trộn thông tin giữa token; feed-forward xử lý từng vị trí qua projection
và activation. Tôi ví attention là “hỏi các token khác”, còn FFN là “suy nghĩ tại
chỗ sau khi đã lấy thông tin”. Hai phần bổ sung chứ không thay thế nhau.

## Encoder, decoder và causal LM

Encoder có thể nhìn toàn bộ input, hợp cho representation. Decoder causal chỉ
nhìn quá khứ, hợp cho sinh token từng bước. LLM chat thường dùng decoder-only,
nhưng checkpoint/model card phải quyết định contract chứ không đoán từ tên.

## Training và inference

Training xử lý batch và backprop; inference thường sinh từng token, giữ KV cache
để không tính lại key/value của prefix. Batch, sequence length, precision và
sampling ảnh hưởng latency/RSS.

Context window là giới hạn sequence, không phải lời hứa câu trả lời sẽ nhớ mọi
chi tiết. Attention có thể bị loãng khi context dài, nên retrieval và prompt
budget vẫn quan trọng.

## Bẫy khi giải thích Transformer

- attention weight không phải causal explanation tuyệt đối;
- token probability không phải xác suất sự thật;
- tăng parameter không tự sửa data leakage;
- quantization không chỉ là giảm file size, nó đổi precision;
- context length khác output token limit;
- benchmark tiếng Anh không đại diện hoàn toàn cho tiếng Việt.

## Cách tôi kiểm một model

1. đọc model card và tokenizer contract;
2. kiểm chat template/system message;
3. thử context nhỏ trước context dài;
4. đo prompt processing và generation riêng;
5. so output deterministic ở temperature 0;
6. chạy held-out câu tiếng Việt và câu code-mix;
7. lưu model revision, quantization, backend và hardware.

## Liên hệ với RAG

RAG đưa evidence vào context, nhưng Transformer chỉ thấy token. Nó không biết
đoạn nào là instruction nếu prompt không phân tách; vì vậy retrieved note phải
được đánh dấu untrusted và system prompt phải nói rõ authority.

## Tôi còn muốn hiểu thêm

- long-context attention và memory compression;
- RoPE scaling có giới hạn gì ngoài việc đổi theta;
- activation sparsity ảnh hưởng serving thế nào;
- interpretability nên dùng để debug hay chỉ nghiên cứu;
- cách đo factuality tách khỏi style fluency.

## Tóm tắt

Attention là cơ chế trộn thông tin có điều kiện giữa các token. Transformer mạnh
vì nhiều lớp representation, không phải vì một công thức attention đơn lẻ. Tôi
luôn tách kiến trúc, objective, data, decoding và serving khi giải thích hành vi.

## Q, K, V theo cách tôi hình dung

Mỗi token tạo query, key và value. Query hỏi “tôi cần thông tin kiểu gì?”, key
cho biết token này có thể match với câu hỏi nào, value là nội dung được trộn vào.
Score từ QK quyết định trọng số; softmax biến score thành phân phối. Đây là mô
hình hóa trực giác, không có nghĩa model thực sự có một câu hỏi bằng ngôn ngữ tự
nhiên trong mỗi head.

Self-attention cho token nhìn token trong cùng sequence, còn causal mask chặn
nhìn token tương lai khi autoregressive generation. Quên mask có thể làm training
loss đẹp nhưng inference behavior sai vì lúc sinh token tương lai chưa tồn tại.

## Multi-head và representation

Một head có thể học dependency gần, head khác học pattern dài, nhưng không nên
gán diễn giải chắc chắn chỉ từ heatmap. Residual stream, MLP, normalization và
position encoding đều góp phần tạo representation. “Attention map giải thích
model” là hypothesis cần kiểm chứng, không phải ground truth.

## Complexity và context

Attention đầy đủ thường tốn theo bình phương sequence length ở score matrix. Tăng
context không chỉ tăng token input; còn tăng memory, KV cache và latency decode.
Các phương pháp sparse/linear attention đổi trade-off chứ không xóa giới hạn.

Trong SoCa, tôi muốn context budget bao gồm system, profile, retrieved evidence,
working summary, recent turns và output reserve. Model có cửa sổ 32K không đồng
nghĩa tôi được nhét 32K raw note mà không có latency/cost.

## Training khác decoding

Teacher forcing cho model xem token target trong training; generation phải tự dùng
token mình vừa sinh. Exposure bias, sampling, temperature, top-p và stop sequence
ảnh hưởng output. Một prompt tốt nhưng max output quá nhỏ có thể tạo câu rỗng hoặc
truncated; đó là ops/config failure, không nên đổ hết cho architecture.

## Câu hỏi tôi dùng để đọc một paper/model card

- training objective và data mix là gì;
- context length được train hay chỉ extrapolate;
- tokenizer tiếng Việt/code-mix ra sao;
- benchmark có contamination không;
- latency/memory đo trên hardware nào;
- license cho phép use case của mình không;
- failure cases và safety eval có được công bố không.

## Bài tập

- tự viết causal mask cho sequence nhỏ và kiểm cell tương lai bị chặn;
- tính shape Q/K/V khi batch, head và sequence thay đổi;
- so sánh KV cache với recompute trên toy decoder;
- tạo prompt cùng context nhưng đổi output reserve;
- kiểm một attention visualization bằng intervention thay vì tin heatmap;
- đo context length tăng ảnh hưởng TTFT và decode thế nào.
