---
type: learning_note
domain: systems
topic: operating-systems-and-memory
status: active
created: 2026-07-16
updated: 2026-07-28
tags: [systems, os, process, thread, memory, io, cache]
source_kind: personal-study-simulation
---

# Systems: process, memory và I/O — cách tôi hình dung một chương trình chạy

## Điều tôi từng hiểu sai

Tôi từng nghĩ chạy một chương trình là CPU đọc từng dòng code, làm xong rồi trả
kết quả. Góc nhìn đó bỏ qua scheduler, virtual memory, kernel, file descriptor,
cache và những lúc chương trình đang chờ chứ không hề dùng CPU.

Giờ tôi hình dung một process như một “người thuê” có không gian địa chỉ riêng,
file descriptor và trạng thái. Thread là những người cùng ở một căn hộ: dùng
chung heap nhưng mỗi người có stack và program counter riêng. Kernel là người
quản lý tòa nhà, quyết định ai được dùng CPU, ai bị chặn và ai được nhìn thiết bị.

## Process và thread

Process có:

- address space và page table;
- code, data, heap và stack;
- file descriptor table;
- signal và trạng thái scheduling;
- một hoặc nhiều thread.

Thread chia sẻ heap và file descriptor nhưng có stack riêng. Vì chia sẻ nên giao
tiếp nhanh hơn process, nhưng race condition dễ xuất hiện. Tôi không gọi “thread
nhanh hơn process” như một chân lý tuyệt đối; chi phí lock, cache contention và
context switch có thể đảo kết quả.

## User mode và kernel mode

Ứng dụng không tự đọc disk hoặc chỉnh page table. Nó gọi system call, chuyển sang
kernel mode, kernel kiểm tra quyền rồi thao tác thiết bị hoặc tài nguyên. Chuyển
mode có chi phí, nhưng phần chậm thường nằm ở disk/network và queue chứ không chỉ
ở instruction chuyển mode.

Khi debug một lệnh “đọc file”, tôi tách:

1. path resolve và permission;
2. page cache có dữ liệu chưa;
3. syscall có block không;
4. filesystem có chờ disk không;
5. bytes có được copy qua boundary user/kernel không;
6. parser của ứng dụng có tốn thời gian hơn I/O không.

## Virtual memory

Virtual address cho mỗi process một không gian có vẻ liên tục. Page table ánh xạ
virtual page sang physical frame. Vì vậy hai process có thể cùng dùng địa chỉ
ảo giống nhau nhưng không nhìn thấy dữ liệu của nhau.

Nếu page chưa có trong RAM, page fault xảy ra. Page fault không phải lúc nào cũng
là bug; demand paging là cơ chế bình thường. Nhưng major page fault kéo theo disk
I/O sẽ làm latency tăng mạnh.

Tôi phân biệt:

- resident set: các page của process đang ở RAM;
- virtual size: không gian đã map, có thể chưa chiếm RAM;
- shared memory: vùng được nhiều process dùng chung;
- anonymous memory: heap/stack không có file backing;
- mmap file: map nội dung file vào address space.

RSS lớn không tự động nghĩa leak; leak là memory vẫn tăng vì reference còn giữ
những object không còn cần. Ngược lại, RSS nhỏ cũng không chứng minh ứng dụng
khỏe nếu bị swap liên tục.

## Stack, heap và allocation

Stack phù hợp frame có lifetime lồng nhau; heap phù hợp object sống lâu hơn một
call. Allocation nhiều object nhỏ gây fragmentation, metadata overhead và áp lực
garbage collector. Một hot loop làm nhiều string nối có thể chậm vì copy nhiều
lần dù Big-O nhìn vẫn O(n).

Tôi luôn hỏi object có cần tồn tại lâu không, có thể reuse buffer không và có thể
stream thay vì đọc toàn bộ vào RAM không. Trong RAG, đọc toàn bộ vault mỗi query
là cách dễ viết nhưng không phải cách đúng khi corpus lớn.

## Cache hierarchy

CPU cache gần core thì nhanh và nhỏ; RAM lớn hơn nhưng chậm hơn; disk/network
chậm hơn nhiều. “Cùng số operation” không có nghĩa cùng thời gian nếu một bên
đụng cache còn bên kia random access.

Tôi dùng ví dụ array/hash để nhớ locality: array liền nhau thường tận dụng cache,
linked list có thể có pointer chase. Hash table cần cân bằng lookup nhanh với
layout memory; benchmark trên laptop nên ghi cả dataset size và distribution.

## I/O và backpressure

Một chương trình async không làm disk nhanh hơn; nó chỉ cho phép làm việc khác
trong lúc chờ. Nếu producer đọc nhanh hơn consumer xử lý, queue tăng vô hạn nếu
không có backpressure.

Các dấu hiệu cần nhìn:

- queue depth tăng;
- worker idle nhưng request vẫn chậm vì lock;
- CPU thấp nhưng iowait cao;
- RSS tăng theo queue;
- p99 tăng dù p50 ổn.

## Cách tôi debug latency

1. đo từ entry đến exit bằng monotonic clock;
2. chia stage CPU, lock, filesystem, network và model;
3. xem p50/p95/p99 thay vì chỉ mean;
4. ghi cold start riêng warm path;
5. đo RSS trước/sau và peak trong lúc build index;
6. kiểm tra contention bằng profile, không đoán từ cảm giác.

## Câu hỏi follow-up tôi dùng để tự kiểm

- process bị OOM là do heap, page cache hay queue giữ object?
- model load chậm do disk, mmap, compilation hay warm-up?
- lock đang bảo vệ invariant nào, có thể giảm vùng critical section không?
- async task bị block bởi hàm sync nào?
- latency spike có trùng GC, page fault hoặc generation swap không?

## Tóm tắt kiểu của tôi

OS là lớp biến một đống tài nguyên hữu hạn thành cảm giác mỗi chương trình có
không gian riêng. Khi hệ thống chậm, tôi không chỉ nhìn code; tôi lần theo đường
đi của dữ liệu qua CPU, cache, RAM, kernel, disk/network và queue.

## Virtual memory không phải RAM miễn phí

Mỗi process thấy virtual address space, nhưng page có thể đang ở RAM, file mapping
hoặc swap. Page fault nhẹ có thể chỉ cần map page; page fault nặng phải đọc disk
và làm latency nhảy vọt. Vì vậy “process dùng 2 GB virtual memory” không đồng
nghĩa đã chiếm 2 GB resident RAM, nhưng cũng không nên coi con số đó vô nghĩa.

Memory mapping giúp load model hoặc index hiệu quả ở vài workload, nhưng file
mapping không làm dữ liệu private tự động. Permission file, lifetime mapping và
behavior khi file bị thay đổi vẫn là contract của ứng dụng.

## Queue là nơi latency ẩn

Một stage có thể xử lý nhanh nhưng queue trước nó tăng dần. Người dùng chỉ thấy
toàn bộ response chậm và tưởng model chậm. Tôi ghi queue depth, enqueue time,
start time, finish time và cancellation. Với voice, audio callback không được
đợi một công việc LLM đồng bộ chạy lâu.

Backpressure là hành vi có chủ đích: drop stale partial, giới hạn pending work,
hoặc báo busy. Không giới hạn queue chỉ làm out-of-memory trễ hơn. Retry cũng tạo
thêm queue nếu không có budget và idempotency.

## CPU, accelerator và false signal

Tên provider hoặc device không chứng minh kernel thực sự chạy ở đó. Tôi muốn xem
selected/fallback provider, operator placement nếu có, memory copy và warm/cold
latency. Một model có thể chạy phần lớn trên accelerator nhưng rơi một operator
xuống CPU khiến p95 xấu.

## Khi debug memory

Tôi phân biệt leak, cache growth, fragmentation và workload tăng. RSS tăng không
đủ để kết luận leak. Tôi chụp snapshot theo thời điểm, xem object/arena, theo dõi
allocation rate và chạy cùng workload sau khi GC/idle.

Với model, KV cache và batch size thường là biến lớn. Tăng context có thể làm
memory tăng theo cách khác với tăng số layer. Tôi đo max resident, startup peak,
steady state và lúc compact/reload.

## Checklist cho một process assistant

1. process nào sở hữu microphone/audio callback;
2. thread nào được phép block;
3. file descriptor/socket nào mở;
4. cancellation có tới được worker không;
5. model/index có unload được không;
6. error path có đóng file/temp resource không;
7. metrics có phân biệt cold start và steady state không.

## Failure cases

- worker chết nhưng UI vẫn hiện spinner;
- callback bị block khi network request treo;
- mmap giữ file generation cũ nên delete không giải phóng disk;
- context lớn làm allocator peak vượt steady-state limit;
- retry nhân đôi request và memory queue;
- process con giữ handle khiến parent không exit sạch;
- permission artifact đúng lúc tạo nhưng sai sau copy/migration.

## Tóm tắt mở rộng

Tôi hình dung OS như người quản lý kho: CPU, RAM, disk và network có quota; cache
giúp lấy nhanh nhưng không phải nguồn sự thật; queue giữ công việc nhưng có thể
trở thành nợ. Khi SoCa chậm, tôi tìm stage và resource boundary bằng trace thay
vì tăng timeout mù.
