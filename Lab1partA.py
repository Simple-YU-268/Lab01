import sys
import heapq
import random
import matplotlib.pyplot as plt


# ========== 全局配置 ==========
MEAN_PKT_SIZE_BITS = 10000               # 平均包大小（比特）
LINK_RATE_BPS = 10_000_000_000           # 链路速率（比特/秒）

# ========== 事件定义 ==========
class Event:
    def __init__(self, time, type, pkt_id):
        self.time = time
        self.type = type  # 'arrival' or 'departure'
        self.pkt_id = pkt_id

    def __lt__(self, other):
        return self.time < other.time  # heapq 使用这个来排序

# ========== 模拟主类 ==========
class MM1Simulator:
    def __init__(self, npkts, lambd):
        self.npkts = npkts
        self.lambd = lambd  # 到达率 λ，单位为 包/μs
        self.queue = []
        self.event_list = []
        self.curr_time = 0.0
        self.next_pkt_id = 0
        self.in_service = False
        self.total_delay = 0.0
        self.total_pkts = 0
        self.queue_len_record = [0] * 12  # P(0) 到 P(10) + P(n>10)
        self.pkts_in_system = 0
        self.area_under_q = 0.0
        self.last_event_time = 0.0

    def schedule_event(self, event):
        heapq.heappush(self.event_list, event)

    def run(self):
        first_arrival_time = random.expovariate(self.lambd)
        self.schedule_event(Event(first_arrival_time, 'arrival', self.next_pkt_id))
        self.next_pkt_id += 1

        while self.total_pkts < self.npkts:
            event = heapq.heappop(self.event_list)
            self.curr_time = event.time
            self.area_under_q += self.pkts_in_system * (self.curr_time - self.last_event_time)
            self.last_event_time = self.curr_time

            if event.type == 'arrival':
                self.handle_arrival(event)
            elif event.type == 'departure':
                self.handle_departure(event)

    def handle_arrival(self, event):
        print(f"[{event.time:.2f}]: pkt {event.pkt_id} arrives and finds {len(self.queue)} packets in the queue")

        index = self.pkts_in_system if self.pkts_in_system <= 10 else 11
        self.queue_len_record[index] += 1
        self.pkts_in_system += 1

        # 安排下一个到达事件
        next_arrival_time = self.curr_time + random.expovariate(self.lambd)
        self.schedule_event(Event(next_arrival_time, 'arrival', self.next_pkt_id))
        self.next_pkt_id += 1

        self.queue.append((event.pkt_id, self.curr_time))

        if not self.in_service:
            self.start_service()

    def handle_departure(self, event):
        pkt_id, arrival_time = self.queue.pop(0)
        delay = self.curr_time - arrival_time
        print(f"[{event.time:.2f}]: pkt {pkt_id} departs having spent {delay:.2f} us in the system")
        self.total_delay += delay
        self.total_pkts += 1
        self.pkts_in_system -= 1

        if self.queue:
            self.start_service()
        else:
            self.in_service = False

    def start_service(self):
        self.in_service = True
        pkt_id, _ = self.queue[0]

        pkt_size_bits = random.expovariate(1 / MEAN_PKT_SIZE_BITS)
        service_time = (pkt_size_bits / LINK_RATE_BPS) * 1e6  # 转为微秒
        departure_time = self.curr_time + service_time

        self.schedule_event(Event(departure_time, 'departure', pkt_id))

    def print_stats(self):
        N = self.area_under_q / self.curr_time
        T = self.total_delay / self.total_pkts
        print(f"\nSimulation Results (λ = {self.lambd}):")
        print(f"Total packets simulated (departed): {self.total_pkts}")
        print(f"Total packets arrived: {self.next_pkt_id}")
        print(f"Average number in system (N): {N:.4f}")
        print(f"Average time in system (T): {T:.4f} us")

        total_arrivals = sum(self.queue_len_record)
        print("\nP(n) distribution (based on arrivals):")
        pn_probs = []
        for n in range(11):
            p = self.queue_len_record[n] / total_arrivals
            pn_probs.append(p)
            print(f"P({n}) = {p:.4f}")
        p_over_10 = self.queue_len_record[11] / total_arrivals
        pn_probs.append(p_over_10)
        print(f"P(n > 10) = {p_over_10:.4f}")

        total_prob = sum(pn_probs)
        print(f"[Check] Total probability = {total_prob:.6f}")

        # 绘制柱状图
        labels = [str(i) for i in range(11)] + ['>10']
        plt.figure(figsize=(10, 6))
        plt.bar(labels, pn_probs)
        plt.xlabel("Number of packets in system (n)")
        plt.ylabel("Probability P(n)")
        plt.title("P(n) Distribution Based on Packet Arrivals")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()


# ========== 主程序入口 ==========
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mm1_simulator.py <npkts> <lambda>")
        sys.exit(1)

    npkts = int(sys.argv[1])
    lambd = float(sys.argv[2])

    sim = MM1Simulator(npkts, lambd)
    sim.run()
    sim.print_stats()
