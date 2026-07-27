# Quyết định dự án — Chọn TTS Valtec ONNX

#life-vault #project #decision #tts #valtec #onnx

Ngày quyết định: 2026-06-01
Slice: life_vault
Provenance: repository_fact — decision record rút từ phần D3.0 trong `BENCHMARKS.md`.

## Quyết định

SoCa chọn Valtec ONNX làm TTS baseline hiện tại, với voice `NF` trong profile
`baseline`.

## Vì sao

1. Đây là runtime tiếng Việt local đã được tích hợp vào đường chạy sản phẩm.
2. Cutover hiện tại dùng bốn ONNX graph fp32 và không cần gửi transcript lên
   cloud.
3. Valtec có số đo E2E đã được ghi trong benchmark, nên phù hợp làm baseline
   để so sánh các ứng viên khác.

## Phạm vi của quyết định

Đây là lựa chọn baseline cho demo/runtime, không phải tuyên bố Valtec luôn có
chất lượng cao nhất. Các ứng viên VieNeu, Piper và runtime khác vẫn có thể được
đánh giá trong bake-off riêng.

## Nguồn

- `BENCHMARKS.md#D3.0 — Valtec ONNX release (current, cutover complete)`
- `soca/tts/registry.py`
