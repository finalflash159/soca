---
type: learning_note
domain: dsa
topic: arrays-and-hash-tables
status: active
created: 2026-07-15
updated: 2026-07-27
tags: [dsa, arrays, hashing, complexity, invariants]
source_kind: personal-study-note
---

# DSA: array, hash table và cách tôi nhìn complexity

## Góc nhìn ban đầu

Tôi từng học Big-O như một bảng thuộc lòng: array là O(1), tìm kiếm là O(n),
hash map là O(1). Sau đó tôi hiểu con số chỉ có nghĩa khi nói rõ thao tác nào,
layout bộ nhớ nào và assumption nào. “O(1)” cũng không có nghĩa chạy nhanh trong
mọi input; constant, cache miss và allocation vẫn ảnh hưởng trải nghiệm thật.

## Array không chỉ là danh sách

Array là một vùng ô nhớ liền nhau. Vì vậy truy cập `a[i]` nhanh: địa chỉ được
tính từ địa chỉ đầu cộng offset. Tôi hình dung nó như một dãy ngăn kéo đánh số;
biết số ngăn thì không cần mở các ngăn trước.

Đổi lại, chèn phần tử vào đầu phải dịch nhiều phần tử. Nếu array dynamic hết
capacity, hệ thống cấp vùng mới, copy dữ liệu rồi giải phóng vùng cũ. Amortized
O(1) cho append không có nghĩa từng lần append đều O(1).

## Invariant đầu tiên

Khi dùng array, tôi luôn ghi:

- index hợp lệ là `[0, n)`;
- `n` là số phần tử thật, capacity có thể lớn hơn n;
- sau mỗi phép xoá, thứ tự còn lại có được giữ không;
- dữ liệu có được phép duplicate không;
- null/sentinel có ý nghĩa gì.

Nhiều bug tưởng là algorithm sai thực ra là invariant index bị phá ở boundary.

## Hash table là gì theo cách tôi hiểu

Hash table biến key thành một số rồi dùng số đó để chọn bucket. Tôi ví nó như
tủ hồ sơ: hash là cách chọn ngăn, còn collision là hai hồ sơ cùng bị chỉ vào một
ngăn. Hash tốt giúp phân bố đều nhưng không xóa được collision.

Collision có thể giải bằng chaining hoặc open addressing. Chaining dễ hiểu nhưng
tạo pointer/đối tượng; open addressing tận dụng array tốt hơn nhưng deletion và
load factor phức tạp hơn.

## Load factor và resize

Khi quá nhiều key vào ít bucket, chain dài hoặc probing dài. Hash table thường
resize trước khi đầy. Resize là bước O(n), nhưng nếu tăng capacity theo cấp số
nhân thì tổng chi phí của nhiều insert vẫn amortized O(1).

Tôi không coi average-case O(1) là cam kết tuyệt đối. Hash collision có thể làm
worst-case O(n), và key do user kiểm soát có thể tạo tình huống xấu nếu hash không
được bảo vệ.

## Mẫu bài toán tôi thường gặp

### Đếm tần suất

Tôi dùng map `value → count`. Điều quan trọng không phải cú pháp map mà là việc
chọn đúng identity của value: normalized string khác raw string và hai object
giống nội dung chưa chắc cùng identity.

### Two sum

Tôi duyệt một lần, lưu các số đã thấy và hỏi map có `target - x` chưa. Invariant
là map chỉ chứa phần tử ở bên trái vị trí hiện tại. Nhờ vậy không dùng chính một
phần tử hai lần.

### Sliding window

Tôi giữ cửa sổ `[left, right]` và invariant cửa sổ luôn hợp lệ sau khi while
đẩy `left`. Hai con trỏ mỗi cái chỉ đi về phía trước nên tổng thao tác là O(n),
không phải O(n²) dù có vòng lặp lồng nhau.

### LRU cache

LRU cần map để tìm node nhanh và doubly linked list để đổi thứ tự gần đây nhanh.
Map trả lời “key ở đâu?”, list trả lời “thứ tự eviction là gì?”. Một cấu trúc
không đủ nếu vừa cần lookup O(1) vừa cần move-to-front O(1).

## Complexity tôi ghi thế nào

| Operation | Array | Hash table average | Câu hỏi thực tế |
| --- | --- | --- | --- |
| index | O(1) | không áp dụng | có cần random access không? |
| search | O(n) | O(1) average | key có hash ổn định không? |
| append | amortized O(1) | O(1) average | resize có làm latency spike? |
| insert đầu | O(n) | không áp dụng | có thể đổi sang deque không? |
| delete | O(n) nếu giữ thứ tự | O(1) average | có cần tombstone không? |

Tôi luôn thêm memory complexity và cost của serialization. Một algorithm nhanh
nhưng tạo hàng triệu object có thể thua algorithm chậm hơn nhưng cache-friendly.

## Bẫy dễ nhầm

- mutable key làm hash thay đổi sau khi insert;
- dùng `in` trên list rồi tưởng là O(1);
- xoá trong khi đang iterate làm skip phần tử;
- dùng hash map khi cần thứ tự ổn định nhưng không ghi rõ contract;
- xem benchmark một input là bằng chứng cho mọi distribution;
- quên Unicode normalization khi key là tiếng Việt.

## Cách tôi debug

1. viết invariant trước khi nhìn implementation;
2. tạo input rỗng, một phần tử và duplicate;
3. thử key collision hoặc hash xấu;
4. kiểm tra boundary `left == right` và `left > right`;
5. đo allocation và memory, không chỉ wall-clock;
6. so implementation với brute-force oracle trên input nhỏ.

## Liên hệ với hệ thống hiện tại

Index knowledge có map từ chunk ID đến chunk để đọc hit nhanh. Nhưng chunk ID,
path và content hash là ba khái niệm khác nhau; không dùng path làm identity duy
nhất nếu rename phải được audit. Citation cần line range, còn hash cần bảo đảm
nội dung đúng generation.

## Bài tự luyện

- implement counter rồi thêm top-k ổn định khi tie;
- viết sliding window cho chuỗi Unicode có dấu;
- thiết kế LRU có giới hạn byte thay vì số entry;
- so sánh hash map với sorted array khi n rất nhỏ;
- chứng minh vì sao two-pointer không quay ngược;
- viết property test cho invariant của dynamic array.

## Tóm tắt theo cách tôi nói

Array mạnh vì dữ liệu nằm gần nhau; hash table mạnh vì biết cách nhảy đến nơi có
key. Muốn chọn đúng cấu trúc, tôi không hỏi “Big-O là gì?” trước mà hỏi “tôi cần
identity, thứ tự, locality hay thao tác nào là hot path?”.

## Worked example: deduplicate nhưng vẫn giữ thứ tự

Nếu input là danh sách event và tôi cần giữ lần xuất hiện đầu tiên, set giúp kiểm
tra membership O(1) trung bình, còn output array giữ order. Tôi không sort chỉ để
deduplicate vì sort phá thứ tự và đổi contract. Nếu key là object, cần một hàm
identity ổn định; stringify tùy tiện có thể phụ thuộc thứ tự field.

Nếu cần đếm tần suất, hash map `key -> count` phù hợp. Nếu cần trả top-k theo
count, map chưa đủ; tôi cần heap hoặc bucket tùy range count. Một cấu trúc không
“tốt” chung chung, nó tốt cho query pattern cụ thể.

## Hashing không miễn phí

Average O(1) dựa trên hash phân bố đủ đều, capacity và collision policy. Trong
worst case, nhiều key có thể va vào cùng bucket. Runtime hiện đại có mitigation,
nhưng tôi không dùng average complexity để bỏ qua input độc hại hoặc key do user
kiểm soát.

Load factor cao làm collision tăng; resize tốn O(n) ở một thời điểm nếu không có
incremental rehash. Khi map là hot path, tôi đo allocation và cache locality chứ
không chỉ đo số phép so sánh.

## Array và cache locality

Array truy cập tuần tự thường hưởng cache locality. Linked list có O(1) insert
đầu khi đã có node nhưng pointer chasing khiến benchmark thực tế kém hơn kỳ vọng.
Nếu workload là append rồi scan, dynamic array thường đơn giản và nhanh. Nếu
workload là random insert giữa hàng triệu phần tử, phải xem lại representation,
không lấy một Big-O đơn lẻ làm quyết định.

## Invariant tôi viết trước khi tối ưu

- prefix `[0:i]` luôn đã được xử lý;
- two-pointer `left <= right` và vùng ngoài pointer đã có classification;
- monotonic stack giữ thứ tự tăng/giảm theo mục tiêu;
- heap đầu luôn là phần tử min/max đúng với comparator;
- map count khớp số item đã consume.

Nếu invariant không nói được bằng câu đơn giản, implementation đang mơ hồ. Tôi
thường viết bản O(n²) rõ ràng trước rồi dùng property test đối chiếu bản tối ưu.

## Các lỗi thực tế

- dùng mutable key trong hash map rồi sửa field sau khi insert;
- dùng `-1` làm sentinel nhưng input hợp lệ cũng có `-1`;
- integer overflow khi cộng prefix sum trên dữ liệu lớn;
- sort in-place làm hỏng list mà caller còn giữ;
- xóa item trong lúc iterate khiến phần tử kế tiếp bị bỏ qua;
- coi empty input và missing input là cùng một trạng thái;
- benchmark warm cache rồi tuyên bố mọi workload đều nhanh.

## Bảng chọn nhanh

| Nhu cầu | Cấu trúc đầu tiên tôi cân nhắc | Câu hỏi phản biện |
| --- | --- | --- |
| giữ order và append | dynamic array | resize/capacity ra sao? |
| lookup theo identity | hash map | key có ổn định và tin cậy không? |
| range query/order | sorted array hoặc tree | update/read tỷ lệ nào? |
| top-k liên tục | heap | k và tie contract là gì? |
| membership cố định | set/bitset | false positive có chấp nhận không? |

## Bài tập tôi dùng để hiểu chứ không chỉ nhớ

- viết hash map nhỏ có collision test;
- so sánh list/set trên input random và adversarial;
- chứng minh two-pointer bằng loop invariant;
- fuzz dynamic array với sequence append/pop/insert/delete;
- benchmark contiguous array với linked structure trên cùng workload;
- kiểm tra duplicate keys, unicode normalization và empty string.
