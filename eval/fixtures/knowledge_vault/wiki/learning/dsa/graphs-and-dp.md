---
type: learning_note
domain: dsa
topic: graphs-and-dynamic-programming
status: active
created: 2026-07-28
updated: 2026-07-29
tags: [dsa, graph, bfs, dfs, shortest-path, dynamic-programming, invariants]
source_kind: personal-study-note
---

# DSA: graph và dynamic programming — tôi tìm cấu trúc trước khi tìm công thức

## Vì sao tôi thêm note này

Tôi từng học graph và DP như hai chương rời rạc: graph có BFS/DFS, DP có công
thức truy hồi. Khi gặp bài mới, tôi thường nhớ sai template rồi cố nhét input vào
template đó. Cách hữu ích hơn là hỏi dữ liệu đang mô tả quan hệ gì, trạng thái nào
được lặp lại, và một lời giải đúng phải giữ invariant nào.

Note này không phải danh sách mẹo thi. Tôi viết lại cách mình đọc đề, cách chọn
biểu diễn, cách chứng minh không bỏ sót trường hợp và cách test những input làm
template quen thuộc thất bại.

## Graph trước hết là cách mô hình hóa

Graph là tập đỉnh và cạnh, nhưng câu hỏi thực tế thường không gọi tên như vậy.
Mạng dependency, đường đi giữa các địa điểm, quan hệ follow, pipeline task và
link giữa các note đều có thể là graph. Trước khi code, tôi ghi rõ:

- đỉnh là gì và có trùng identity không;
- cạnh có hướng hay vô hướng;
- cạnh có trọng số, capacity hoặc thời gian không;
- có được đi lại một đỉnh/cạnh không;
- output cần đường đi, khoảng cách, số thành phần hay thứ tự.

Nếu không ghi năm dòng này, tôi rất dễ dùng BFS cho graph có trọng số hoặc coi
một dependency có thể chạy ngược chiều. Đây là lỗi mô hình hóa, không phải lỗi
syntax.

## Biểu diễn và chi phí thật

Adjacency list phù hợp khi số cạnh thưa: bộ nhớ O(V + E), duyệt hàng xóm tổng
cộng O(V + E). Adjacency matrix tốn O(V²), nhưng kiểm tra một cạnh là O(1) và
có thể hợp lý khi graph nhỏ, dày hoặc cần phép toán ma trận.

Tôi không gọi list luôn tốt hơn matrix. Với một graph 50 đỉnh gần như đầy cạnh,
matrix đơn giản và predictable có thể tốt. Với graph dependency hàng trăm nghìn
đỉnh, matrix là lựa chọn không thể chấp nhận. Complexity chỉ có nghĩa sau khi
biết scale và thao tác nóng.

Tôi thường lưu adjacency list dạng mapping `node -> list[(neighbor, weight)]`.
Nếu identity là string, tôi cần normalize ở biên và không để một node có cả
`"ONNX"` lẫn `"onnx"` nếu domain không phân biệt hoa thường.

## BFS: lớp khoảng cách không trọng số

BFS giống như đổ nước từ source: vòng đầu là mọi đỉnh cách một bước, vòng sau
là hai bước. Queue giữ thứ tự lớp. Khi graph không trọng số, lần đầu tôi gặp một
đỉnh chính là lần có số cạnh ít nhất từ source.

Invariant tôi dùng là: khi lấy node ra khỏi queue, `distance[node]` đã là khoảng
cách ngắn nhất. Để invariant đúng, tôi chỉ enqueue node khi chưa visited. Nếu
đánh dấu quá muộn, hai hàng xóm có thể enqueue cùng một node nhiều lần và code
vẫn “ra đáp án” nhưng tốn tài nguyên bất ngờ.

BFS cũng dùng cho grid. Cell là node, bốn hoặc tám hướng là cạnh. Tôi kiểm tra
boundary trước khi truy cập grid và coi obstacle là không có cạnh. Một lỗi hay
gặp là đánh dấu visited sau khi dequeue, tạo queue phình rất lớn trên vùng mở.

## DFS: khám phá, không mặc định là đường ngắn nhất

DFS đi sâu trước, hợp để tìm connected component, cycle, topological ordering
và các bài cần trạng thái vào/ra. Nhưng DFS không tự cho shortest path. Nếu cần
đường ít cạnh nhất, tôi phải dùng BFS hoặc thuật toán có trọng số phù hợp.

Với graph có hướng, cycle detection bằng ba trạng thái dễ nói hơn màu trắng/xám/
đen: chưa vào, đang ở recursion stack, đã hoàn tất. Gặp cạnh đến node đang ở
stack là back-edge và có cycle. Chỉ thấy node đã visited không đủ để kết luận
cycle vì cạnh đến một nhánh hoàn tất có thể hoàn toàn hợp lệ.

Topological sort chỉ tồn tại khi graph có hướng không cycle. Kết quả không nhất
thiết duy nhất. Vì vậy test nên kiểm tra mọi cạnh `u -> v` đều có `order[u] <
order[v]`, không hard-code đúng một thứ tự.

## Shortest path: chọn theo mô hình trọng số

Tôi dùng BFS cho cạnh cùng cost. Dijkstra dùng khi trọng số không âm; priority
queue lấy candidate nhỏ nhất, và relaxation thử cập nhật `dist[v]` qua `u`.
Bellman-Ford chậm hơn nhưng chịu được cạnh âm và phát hiện negative cycle.

Không được dùng Dijkstra chỉ vì tên quen thuộc. Một cạnh âm có thể làm node đã
được “chốt” trở nên tốt hơn sau đó. A* thêm heuristic để tìm nhanh trong không
gian có ước lượng khoảng cách, nhưng heuristic cần admissible nếu muốn giữ bảo
đảm tối ưu.

Trong hệ thống thật, weight có thể là latency, cost hoặc risk chứ không phải
độ dài. Tôi phải ghi rõ đơn vị và cách cộng; tối ưu latency p95 khác với tối ưu
chi phí trung bình.

## Dynamic programming: trạng thái là phần khó nhất

DP không phải “thấy lặp thì memoize”. Tôi cần định nghĩa state đủ để phần còn lại
của bài không phụ thuộc lịch sử đã bị bỏ. Nếu state thiếu thông tin, hai prefix
khác nhau có thể bị gộp dù tương lai của chúng khác nhau; khi đó cache trả lời
sai rất thuyết phục.

Ví dụ knapsack 0/1: `dp[i][w]` là giá trị tốt nhất khi chỉ xét i món đầu và sức
chứa còn là w. Transition chọn không chọn món i. Tôi không được dùng lại cùng
món nhiều lần trừ khi bài là unbounded knapsack và thứ tự vòng lặp phản ánh điều
đó.

LCS dùng state theo hai prefix. Khi ký tự cuối giống nhau, tôi đi vào hai prefix
ngắn hơn; khi khác, lấy max của bỏ một đầu. Điều quan trọng không phải thuộc công
thức mà là chứng minh mọi solution tối ưu rơi vào một trong các nhánh transition.

## Tối ưu state và thứ tự vòng lặp

Nếu `dp[i]` chỉ phụ thuộc `dp[i-1]`, tôi có thể nén còn hai row. Nếu nén một row,
thứ tự cập nhật quyết định item được dùng một hay nhiều lần. Cập nhật capacity
giảm dần thường giữ 0/1 constraint; tăng dần thường cho phép reuse. Đây là một
chỗ tôi từng sửa “cho pass” mà không hiểu nên bug chỉ lộ ở input nhiều món.

Tôi ghi dependency graph của state trước khi nén bộ nhớ. Nếu không làm vậy,
optimization biến thành thay đổi semantics. Test của bản nén phải so với bản
tham chiếu chậm trên input random nhỏ, không chỉ vài ví dụ hand-written.

## Khi graph và DP gặp nhau

Shortest path trên DAG là DP theo topological order. Memoized DFS trên graph cũng
là DP nếu graph state không có cycle hoặc cycle đã được xử lý. State machine,
planning và parsing đều có thể nhìn như graph các trạng thái.

Góc nhìn này giúp tôi không phân loại bài quá sớm. Tôi hỏi: có thể đi từ state
này sang state kia không, cost là gì, thứ tự dependency ra sao, và có subproblem
trùng không? Cùng một bài có thể giải bằng BFS, Dijkstra hoặc DP tùy objective.

## Test và failure cases tôi luôn thêm

- graph rỗng, một node, nhiều component;
- self-loop, parallel edge, cycle và DAG;
- weight bằng 0, weight rất lớn, cạnh âm khi thuật toán không hỗ trợ;
- grid toàn obstacle, source trùng target, source ngoài boundary;
- DP capacity bằng 0, item weight lớn hơn capacity, duplicate item;
- input có identity khác kiểu hoặc thứ tự cạnh bị đảo;
- graph lớn dạng chain để bắt recursion depth;
- tie nhiều lời giải tối ưu để kiểm contract thay vì ép một output.

Tôi dùng brute force trên input nhỏ để đối chiếu shortest path/knapsack. Property
test kiểm invariant hơn là chỉ kiểm một output. Nếu thuật toán có randomization,
seed phải nằm trong trace để reproduce.

## Tóm tắt theo cách tôi nói

Graph giúp tôi nhìn quan hệ và đường đi; DP giúp tôi lưu kết quả của state đã
định nghĩa đúng. Template chỉ là implementation sau khi mô hình hóa. Muốn code
đúng, tôi phải nói rõ invariant, giới hạn của thuật toán, và một input mà nó
không được phép xử lý. Cảm giác “đã nhớ công thức” không thay thế được bước đó.

## Câu hỏi tôi còn muốn tự kiểm tra

- khi graph thay đổi liên tục, có nên incremental shortest path hay rebuild;
- cách chọn heuristic A* khi metric có nhiều mục tiêu;
- làm sao visualize state DP để phát hiện state bị thiếu;
- benchmark adjacency list/matrix trên workload thật thay vì đo microbenchmark;
- khi nào dùng thư viện graph thay vì tự giữ implementation trong product.

## Worked example: dependency của các note

Giả sử mỗi learning note có thể link sang prerequisite. Tôi tạo cạnh
`prerequisite -> note`, rồi topological sort để gợi ý thứ tự học. Nếu có cycle
giữa A và B, đó không nhất thiết là lỗi dữ liệu; có thể hai note tham chiếu lẫn
nhau. Nhưng hệ thống phải báo cycle thay vì đưa một thứ tự giả chắc chắn.

Nếu tôi muốn tìm ít bước nhất giữa hai khái niệm, cạnh không trọng số và BFS đủ.
Nếu “chi phí học” khác nhau theo độ khó, tôi cần weight và phải nói cost là gì.
Query “liên quan nhất” lại là retrieval/ranking, không nên nhầm thành shortest
path trong graph link.

## Worked example: chia state DP

Với một chuỗi token, tôi có thể định nghĩa state `(position, mode)` nếu tương lai
chỉ phụ thuộc vị trí và mode. Nếu còn phụ thuộc token trước đó nhưng state bỏ đi,
hai history khác nhau bị gộp sai. Tôi viết hai history dẫn tới cùng state và hỏi
“phần còn lại có thật sự giống nhau không?” trước khi memoize.

## Độ đúng và performance

Tôi đo V/E, số state, memory peak và số transition, không chỉ input length. Graph
thưa/dày có thể đổi algorithm phù hợp. Memoization có thể giảm thời gian nhưng
đổi memory; iterative DP tránh recursion depth nhưng cần đúng order.

## Test adversarial

- graph có 100.000 node thành một chain;
- nhiều cạnh trùng và self-loop;
- weight bằng nhau tạo nhiều shortest path;
- DP state có giá trị âm và sentinel hợp lệ;
- input unicode/string identity có normalization khác nhau;
- random graph so với brute force trên size nhỏ;
- thay thứ tự input và kiểm output semantics không đổi.

## Tóm tắt cuối

Tôi không chọn BFS/DFS/DP vì tên bài. Tôi bắt đầu bằng object, relation, state,
transition và invariant. Sau đó mới cân nhắc complexity, memory, recursion,
parallelism và cách test. Đó là cách chuyển từ “nhớ thuật toán” sang “hiểu lúc
nào thuật toán đúng”.
