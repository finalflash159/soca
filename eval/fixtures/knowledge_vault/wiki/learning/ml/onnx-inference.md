---
type: learning_note
domain: ml-systems
topic: onnx-runtime-inference
status: active
created: 2026-07-26
updated: 2026-07-28
tags: [onnx, inference, coreml, cpu, model-serving, debugging]
source_kind: personal-study-note
---

# Inference ONNX: tôi kiểm model chạy thật chứ không nhìn tên provider

## Câu hỏi thực tế

Tôi muốn biết model có chạy đúng trên máy local hay không, provider nào nhận
graph, output có đúng shape và latency có chấp nhận được. “Import package thành
công” chỉ là bước đầu; “session tạo được” cũng chưa chứng minh mọi node chạy ở
accelerator.

## Pipeline tôi ghi lại

```text
artifact fingerprint
  → load graph
  → inspect input/output
  → discover available providers
  → create session theo thứ tự ưu tiên
  → warm-up
  → run known input
  → verify dtype/shape/value range
  → record provider + latency + RSS
```

Trên Apple Silicon, tôi có thể ưu tiên `CoreMLExecutionProvider` nếu graph hỗ
trợ, sau đó fallback `CPUExecutionProvider`. Provider có trong danh sách không
nghĩa mọi node đều chạy ở nó; cần xem warning partition/fallback.

## Input contract

Adapter phải biết sample rate, channel, dtype, layout, batch và dynamic axis.
Whisper-like model thường nhận tensor khác TTS model. Sai dtype đôi lúc không
crash ngay mà làm output vô nghĩa, nên smoke phải kiểm range và shape.

## Warm-up và đo latency

Lần đầu có thể tốn load, graph compile, memory mapping hoặc kernel cache. Tôi tách:

- cold process: từ process đến output đầu;
- warm first: sau model load;
- steady state: nhiều input liên tiếp;
- peak RSS trong cold và steady;
- output quality trên câu thật.

Không so một cold CPU với một warm CoreML rồi kết luận provider.

## Bảng failure

| Triệu chứng | Điều tôi kiểm |
| --- | --- |
| model file thiếu | digest, path, permission và download status |
| session không tạo | opset, operator, provider order |
| provider fallback | available list, partition warning |
| output shape sai | adapter contract và dynamic axes |
| latency spike | warm-up, page fault, thread count, input length |
| âm thanh sai | sample rate, normalization, clipping, chunk boundary |

## Cách đọc status

Tôi muốn status nói `ready:coreml+cpu-fallback`, `ready:cpu` hoặc
`degraded:model-missing`, không chỉ `baseline ok`. Model harness, provider,
execution backend và fallback phải hiện riêng vì chúng là failure boundary khác
nhau.

## Ghi chú kết quả chưa đủ benchmark

Fixture này không chứa weight hoặc claim benchmark release. Các smoke table có thể
ghi input class và expected contract, còn số latency phải lấy từ máy thật khi
chạy eval. Tôi không đặt số giả vào note rồi gọi đó là đo.

## Tóm tắt

Inference đúng là một chuỗi contract từ artifact đến output. Provider name chỉ là
metadata; bằng chứng thật là session, selected/fallback provider, output contract
và đo trên warm/cold path.

## Tôi kiểm tra model artifact trước provider

Đầu tiên tôi kiểm input names, dtype, dynamic axes, opset, output shape và
pre/post-processing. Một session tạo được không có nghĩa tokenizer, normalization
hoặc sample rate đúng. Nhiều lỗi “model không tốt” thực ra là input scale/shape
sai.

Tôi giữ model revision/hash và environment trong trace. Nếu chỉ ghi filename,
người khác không biết đã chạy checkpoint nào. Nếu model tải từ remote, cache
path và permission cũng là một phần của reproducibility.

## Provider fallback cần quan sát

Danh sách provider được đăng ký không phải danh sách operator thực sự chạy. Tôi
ghi selected provider nếu runtime cung cấp, fallback, warning và số lần copy CPU.
Nếu provider accelerator không nhận một op, output vẫn có thể đúng nhưng latency
hoặc RAM khác hẳn. Đây là lý do phải đo cả correctness và performance.

## Contract output

Tôi kiểm số output, shape, dtype, finite values, range và semantic sanity trên
sample nhỏ. Với ASR, output text không được chứa token đặc biệt chưa decode. Với
embedding, dimension và normalization phải nhất quán. Với audio, sample rate,
channel và amplitude cần được kiểm.

## Benchmark có warm-up

Tôi tách download/load/session initialization, first inference và steady-state.
Mỗi stage có thể là bottleneck khác nhau. Đo một lần rồi báo latency là không đủ;
tôi dùng nhiều repetition, ghi p50/p95 và peak memory nếu có thể.

## Khi provider không chạy

Tôi không âm thầm tuyên bố “CoreML active” chỉ vì package có CoreML. Nếu fallback
sang CPU, status phải nói rõ. Nếu fallback không đạt latency contract, chọn fail
fast hoặc hướng dẫn cài dependency thay vì treo worker.

## Regression checklist

- model hash và opset không đổi ngoài ý muốn;
- input contract pass trên batch 1 và edge shape;
- output ổn định giữa provider trong tolerance;
- cold/warm latency được ghi riêng;
- provider fallback hiện đúng ở status;
- memory release khi engine shutdown;
- error không in path/secret nhạy cảm.
