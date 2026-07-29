# Checklist trước khi release

- chạy unit và integration suite;
- chạy smoke với local model đã provision;
- chạy remote smoke khi provider key được user cho phép;
- kiểm tra tool call, evidence status, citation và prompt manifest;
- xác nhận no-answer không trả hit lạc đề như bằng chứng;
- kiểm tra chat và voice dùng cùng provider/model setting;
- ghi model, commit, corpus class, hardware và kết quả vào artifact.

Demo smoke có thể dùng showcase vault này. Quality benchmark phải dùng corpus
độc lập hoặc private release đã gắn provenance, không dùng lại demo corpus.
