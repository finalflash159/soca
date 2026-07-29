---
type: learning_note
domain: systems
topic: databases-and-indexes
status: active
created: 2026-07-28
updated: 2026-07-29
tags: [systems, database, sqlite, index, transaction, consistency, query-plan]
source_kind: personal-study-simulation
---

# Systems: database và index — dữ liệu bền vững không chỉ là ghi vào file

## Tôi từng đánh đồng persistence với durability

Tôi từng thấy câu lệnh insert trả về thành công rồi nghĩ dữ liệu đã an toàn. Thực
ra có nhiều lớp: process đã nhận request, engine đã đưa data vào memory, journal
đã ghi, transaction đã commit và filesystem đã flush. Mỗi lớp có một contract
khác. Khi thiết kế index cho SoCa, tôi phải biết metadata nào có thể rebuild và
metadata nào mất là mất dữ liệu.

Một cache vector có thể tạo lại từ Markdown; một proposal memory đã được user
approve thì không nên coi như cache. Cùng là một file SQLite nhưng lifecycle và
permission của từng bảng cần được nói rõ.

## Transaction là ranh giới quan sát

Transaction giúp nhóm nhiều thay đổi thành một đơn vị: người đọc thấy trạng thái
trước hoặc sau, không thấy nửa chừng nếu isolation phù hợp. Tôi không dùng
transaction như từ thần chú. Cần biết rollback bao phủ điều gì, trigger nào chạy,
và file ngoài database có cùng atomicity không.

Ví dụ cập nhật tài liệu: đổi nội dung Markdown, cập nhật manifest và thay vector.
Nếu ba bước không atomically coordinated, process chết giữa chừng có thể để
manifest nói vector mới nhưng file vector vẫn cũ. Thiết kế an toàn hơn là ghi
generation mới vào staging, fsync/close theo contract, rồi đổi một pointer/index
metadata cuối cùng.

## SQLite phù hợp ở đâu

SQLite hợp với một assistant desktop vì không cần daemon, backup đơn giản và
transaction mạnh. Nó phù hợp cho vài nghìn hoặc vài triệu record tùy workload,
nhưng không tự biến thành vector database phân tán. Tôi phải đo kích thước, read
concurrency, write contention và startup.

Index SQLite cho chunk nên có các cột phục vụ lookup thật: vault id, relative
path, content digest, chunk id, line range, updated timestamp và generation. Tôi
không index mọi cột theo cảm giác; mỗi index làm write và disk footprint tăng.

Metadata chứa nội dung note phải có permission private. Quy tắc `0644` có thể biến
cache cá nhân thành dữ liệu người dùng đọc được trên máy shared. Permission phải
được đặt ngay lúc tạo và kiểm tra khi migrate, không chờ người dùng phát hiện.

## Query plan thay cho đoán

Khi query chậm, tôi chạy `EXPLAIN QUERY PLAN` trước. Full table scan có thể đúng
với table nhỏ; thêm index mù quáng không chắc nhanh hơn. Tôi xem cardinality,
selectivity, sort và số row thực tế. Một query có index nhưng phải đọc quá nhiều
row vẫn có thể chậm.

Search lexical của vault có thể có inverted map riêng vì tokenization và ranking
khác SQL B-tree. Tôi không gộp hai loại index thành một thuật ngữ “index”. B-tree
giỏi equality/range/order; inverted index giỏi term-to-document; dense matrix giỏi
similarity exact trên corpus nhỏ.

## Inverted index và vector index

Inverted index map token đến posting list. Khi document mới, chỉ posting của token
đổi; khi sửa document, cần bỏ generation cũ rồi thêm generation mới. Nếu không
remove đúng, một query có thể trả cùng một chunk hai lần hoặc trả stale text.

Dense exact search tính dot product/cosine với mọi vector. Với showcase vài chục
chunk, đây là lựa chọn dễ kiểm tra nhất. FAISS/HNSW chỉ có ý nghĩa khi corpus và
latency yêu cầu; ANN thêm recall/maintenance trade-off. Tôi muốn benchmark recall
trước khi dùng thư viện nặng chỉ vì tên phổ biến.

Vector index cần lưu model id, dimension, normalization, source digest và chunk
generation. Chỉ lưu file `.npy` không đủ để biết vector được sinh bởi model nào.
Nếu dimension hoặc preprocessing đổi, phải rebuild; không cố ghép vector khác
schema vào cùng matrix.

## Incremental update và crash recovery

Digest của raw document giúp phân biệt mtime đổi nhưng nội dung không đổi. Với
chunking ổn định, chunk chưa đổi có thể reuse vector. Nếu chunk boundary trôi
do thêm text ở đầu file, nhiều chunk có thể đổi dù ý cũ gần như giữ nguyên; đó là
trade-off cần đo, không hứa “chỉ embed một dòng”.

Tôi thích manifest có `status: ready|building|failed`, generation, source digest
và error cuối. Startup thấy generation building thì không âm thầm dùng nửa index;
nó rollback về generation ready hoặc rebuild. Một lock file có owner/pid/time giúp
phân biệt process chết với job đang chạy.

Xóa document cũng là update có chủ đích. Tombstone hữu ích khi có watcher nhận sự
kiện trễ, nhưng tombstone phải được compact sau khi chắc không còn reader dùng
generation cũ. “Xóa file nguồn” không nên làm record biến mất mà không trace.

## Backup và phân loại dữ liệu

Tôi chia dữ liệu thành raw source, derived sparse index, derived dense vectors,
runtime cache và user-approved memory. Raw source và approved memory cần backup
theo policy; dense vector có thể rebuild nhưng rebuild time vẫn là cost vận hành.

Backup encrypted không đồng nghĩa mọi consumer được phép đọc. Log lỗi không nên
in chunk text hoặc API key. Snapshot phải ghi schema/version và restore test phải
được chạy định kỳ; file copy chưa chứng minh SQLite snapshot nhất quán.

## Các failure case tôi ghi vào test

- kill process giữa lúc build generation mới;
- hai indexer cùng vault;
- file đổi mtime nhưng digest không đổi;
- file rename, file delete và path traversal;
- model embedding đổi dimension;
- database permission quá rộng;
- stale manifest trỏ vector generation đã bị xóa;
- query database trả duplicate chunk;
- backup restore thiếu metadata nhưng vẫn báo thành công.

## Migration schema

Tôi version schema và viết migration có thể chạy lại hoặc fail rõ. Một migration
đổi tên field cần biết reader cũ còn chạy không; nếu app update trước database,
compatibility window phải được thiết kế. Không sửa file SQLite bằng tay rồi gọi
đó là migration có thể reproduce.

Derived index có thể rebuild, nên migration an toàn thường giữ raw source và tạo
generation mới. Tôi không xóa generation cũ ngay nếu reader đang dùng; garbage
collector cần age/lease hoặc refcount rõ ràng.

## Concurrent readers và writers

Desktop app vẫn có thể có index worker, chat worker và `/status` đọc cùng lúc.
Read path cần thấy một generation nhất quán. Writer không nên mutate list vector
đang được query. Snapshot/pointer swap đơn giản hơn lock mọi query, nhưng cần test
crash và cleanup.

## Quyết định cho corpus nhỏ

Với fixture vài chục tài liệu, tôi ưu tiên Markdown là source, manifest/index
private và exact/hybrid search dễ inspect. Không thêm database/vector service
chỉ để demo nhìn “production”. Khi cần scale, benchmark read/write/recall/cold
start trước, ghi decision và migration plan.

## Bài tập

- kill writer tại từng bước staging/pointer swap;
- chạy hai reader ở hai generation;
- migrate schema hai lần;
- restore snapshot rồi kiểm row count/digest/vector dimension;
- chmod sai và kiểm startup sửa hoặc cảnh báo;
- xóa file rồi kiểm search không trả stale chunk.

## Tóm tắt kiểu của tôi

Index là một bản đồ dẫn đường, không phải nguồn sự thật. Tôi luôn giữ raw source,
schema và generation đủ để giải thích vì sao một hit xuất hiện. Với corpus nhỏ,
exact và dễ kiểm tra thường có giá trị hơn một stack ANN phức tạp. Khi scale tăng,
tôi mới thay index, nhưng giữ nguyên contract: provenance, atomic update,
permission, rebuild và observability.

## Câu hỏi còn mở

- ngưỡng corpus nào khiến exact dense search không còn đáp ứng latency;
- schema manifest nào giúp migrate mà không downtime;
- cách đo chi phí rebuild và mức stale chấp nhận được;
- backup nào cần mã hóa riêng giữa raw note và derived artifact;
- khi nào SQLite FTS đủ tốt thay cho inverted index tự viết.
