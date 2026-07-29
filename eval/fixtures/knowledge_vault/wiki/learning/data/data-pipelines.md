---
type: learning_note
domain: data-engineering
topic: data-pipelines-and-data-quality
status: active
created: 2026-07-29
updated: 2026-07-29
confidence: medium
tags: [data, pipeline, ingestion, quality, lineage, backfill]
source_kind: personal-study-note
---

# Data engineering: pipeline không chỉ là chạy một script từ đầu đến cuối

## Cách tôi từng nhìn sai

Tôi từng nghĩ pipeline là đọc file, transform, ghi output. Khi có một file hỏng,
tôi sửa parser và chạy lại. Nhưng trong hệ thống thật, câu hỏi quan trọng hơn là:
đã ingest record nào, record nào bị duplicate, output tạo từ version nào, và chạy
lại có tạo kết quả khác hoặc nhân side effect không.

Tôi tách pipeline thành ingestion, validation, transformation, materialization
và serving. Mỗi stage có input/output contract và checkpoint riêng.

## Dòng chảy tôi dùng để thiết kế

```text
source
  → immutable raw event/file
  → schema + quality checks
  → normalized records
  → derived/indexed data
  → query serving
  → lineage + metrics
```

Raw là bằng chứng đầu vào. Derived index có thể rebuild. Nếu chỉ giữ output cuối,
tôi mất khả năng giải thích vì sao một record xuất hiện.

## Schema không chỉ là kiểu dữ liệu

Schema còn nói field bắt buộc, nullability, unit, timezone, identity và version.
`amount: 100` không đủ nếu không biết currency. `date: 07/08` không đủ nếu không
biết locale. Một field đổi từ seconds sang milliseconds có thể làm pipeline vẫn
chạy nhưng mọi metric sai.

Tôi giữ schema version và migration strategy. Thêm field optional thường an toàn
hơn đổi nghĩa field cũ. Nếu breaking change cần, tôi chạy dual-read hoặc backfill
có kiểm tra thay vì overwrite mù.

## Idempotency và duplicate

Một job có thể chạy lại vì crash sau khi ghi output nhưng trước khi ghi checkpoint.
Nếu không có stable event ID hoặc content hash, record sẽ nhân đôi. Tôi muốn mỗi
stage có invariant:

- cùng raw identity không tạo hai logical records;
- cùng input version + code version cho output reproducible;
- retry không nhân side effect;
- checkpoint chỉ advance sau khi output durable;
- output partial không được publish như complete.

## Data quality checks

| Check | Ví dụ | Hành động khi fail |
| --- | --- | --- |
| completeness | field bắt buộc không null | quarantine hoặc reject |
| validity | date/enum/unit hợp lệ | báo producer |
| uniqueness | receipt ID không duplicate | deduplicate theo policy |
| consistency | total ledger khớp dòng | mở correction |
| freshness | source đã cập nhật chưa | cảnh báo stale |
| distribution | giá trị lệch bất thường | review, không tự xóa |

Một check pass không chứng minh data đúng nghĩa. Nó chỉ chứng minh data qua
những contract đã viết. Vì vậy tôi cần sample review và lineage.

## Late data và backfill

Sự kiện có `event_time` và `ingest_time` khác nhau. Một receipt ngày 20 có thể
đến hệ thống ngày 22. Nếu partition chỉ theo ingest time, query tháng 07 có thể
thiếu hoặc phải backfill. Tôi ghi watermark và late-arrival policy.

Backfill phải có phạm vi, code version, dry-run, số record trước/sau và rollback
plan. Không gọi một lần chạy lại toàn bộ là “fix” nếu không biết nó thay đổi gì.

## Lineage

Khi assistant trả một chunk knowledge, tôi muốn biết path, content digest, chunk
range và index generation. Khi ledger tổng hợp, tôi muốn biết những receipt nào
đã cộng. Lineage nối output với input để debug và giải thích, không chỉ để làm UI.

## Pipeline của vault

Với Markdown vault, raw source là file. Manifest theo dõi mtime/size/digest.
Chunker tạo chunk ID từ path, line range và text. Sparse/dense index là derived
artifact. Nếu file sửa, chunk mới phải được embed; chunk không đổi có thể reuse.
Nếu file xóa, generation mới phải loại chunk cũ.

Index không được biến thành source of truth duy nhất. Khi vector mất, pipeline
phải rebuild từ Markdown; khi Markdown mất, vector không đủ để khôi phục toàn bộ
provenance.

## Quan sát pipeline

- records read/accepted/rejected;
- duplicate count;
- schema version;
- raw/derived digest;
- chunks reused/re-embedded;
- generation published;
- duration và peak resource;
- error class và quarantine path.

Metric “job success” quá coarse. Tôi cần biết job success nhưng 12% record bị
reject hay success với zero input vì source path sai.

## Failure cases tôi muốn test

1. process chết sau khi ghi output;
2. file đổi trong lúc index;
3. schema field đổi unit;
4. duplicate event đến hai lần;
5. late record của tháng cũ;
6. vector generation mới không publish được;
7. raw source bị xóa sau khi derived index tạo;
8. query đọc generation cũ khi rebuild đang chạy.

## Cách tôi nhớ

Data pipeline tốt tạo ra trạng thái có thể giải thích, không chỉ một file kết quả.
Tôi cần phân biệt source, derived, checkpoint và evidence; nếu không, RAG sẽ
trả đúng một đoạn nhưng không biết đoạn đó được tạo từ phiên bản nào.

## Câu hỏi còn mở

- khi index có hàng triệu chunk, generation pointer và garbage collection nên ra sao?
- quality gate nào đủ để chặn publish mà không làm mất source mới?
- cần lưu raw snapshot bao lâu cho một vault cá nhân?

## Bài tập

Tạo một file Markdown, index, sửa một heading, xóa file rồi inspect manifest.
Đếm chunk reused/new/deleted. Nếu chỉ thấy “index rebuilt”, pipeline chưa cho tôi
đủ evidence để tin lifecycle.
