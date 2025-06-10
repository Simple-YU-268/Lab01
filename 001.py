import random
import heapq
import matplotlib.pyplot as plt

class Packet:
    def __init__(self, number, arrival_time, size):
        self.number = number
        self.arrival_time = arrival_time
        self.size = size  # in Bytes
        self.departure_time = None

class Event:
    def __init__(self, time, event_type, packet):
        self.time = time
        self.event_type = event_type  # 'arrival' or 'departure'
        self.packet = packet

    def __lt__(self, other):
        return self.time < other.time

class Queue:
    def __init__(self):
        self.buffer = []

    def insert(self, packet):
        self.buffer.append(packet)

    def remove(self):
        return self.buffer.pop(0) if self.buffer else None

    def __len__(self):
        return len(self.buffer)

class Server:
    def __init__(self):
        self.service_rate_bps = 10 * (10 ** 9)
        self.current_time = 0
        self.busy = False

class Simulator:
    def __init__(self, npkts, lambd):
        self.npkts = npkts
        self.lambd = lambd
        self.event_list = []
        self.queue = Queue()
        self.server = Server()
        self.time = 0
        self.packet_count = 0
        self.total_delay = 0
        self.total_packets_in_system = 0
        self.pn_counter = [0] * 11  # For P(n), where n in 0..10
        self.generated_packets = 0

    def schedule_event(self, event):
        heapq.heappush(self.event_list, event)

    def generate_packet(self):
        size = int(random.expovariate(1 / 1250))  # bytes
        inter_arrival_time = random.expovariate(self.lambd)  # in us
        arrival_time = self.time + inter_arrival_time
        packet = Packet(self.packet_count, arrival_time, size)
        self.packet_count += 1
        return packet

    def run(self):
        # Schedule first arrival
        first_packet = self.generate_packet()
        self.schedule_event(Event(first_packet.arrival_time, 'arrival', first_packet))

        while self.packet_count < self.npkts or self.event_list:
            if not self.event_list:
                break

            event = heapq.heappop(self.event_list)
            self.time = event.time

            if event.event_type == 'arrival':
                self.handle_arrival(event)
            elif event.event_type == 'departure':
                self.handle_departure(event)

        self.print_summary()

    def handle_arrival(self, event):
        packet = event.packet
        num_in_system = len(self.queue)
        self.pn_counter[min(num_in_system, 10)] += 1

        if not self.server.busy:
            self.server.busy = True
            service_time = (packet.size * 8) / self.server.service_rate_bps * 1e6  # us
            packet.departure_time = self.time + service_time
            self.schedule_event(Event(packet.departure_time, 'departure', packet))
        else:
            self.queue.insert(packet)

        if self.packet_count < self.npkts:
            new_packet = self.generate_packet()
            self.schedule_event(Event(new_packet.arrival_time, 'arrival', new_packet))

    def handle_departure(self, event):
        packet = event.packet
        delay = packet.departure_time - packet.arrival_time
        self.total_delay += delay
        self.total_packets_in_system += 1

        if len(self.queue) > 0:
            next_packet = self.queue.remove()
            service_time = (next_packet.size * 8) / self.server.service_rate_bps * 1e6  # us
            next_packet.departure_time = self.time + service_time
            self.schedule_event(Event(next_packet.departure_time, 'departure', next_packet))
        else:
            self.server.busy = False

    def print_summary(self):
        avg_delay = self.total_delay / self.total_packets_in_system
        avg_packets = self.total_delay / self.time
        print("Summary:")
        print("-------------------------------------------")
        print(f"Average number of packets in the system N : {avg_packets:.3f}")
        print(f"Average time spent by a packet in the system T : {avg_delay:.3f} us")

        print("Probability P(n) that an arriving packet finds n packets already in the system:")
        total_arrivals = sum(self.pn_counter)
        for n in range(11):
            prob = self.pn_counter[n] / total_arrivals if total_arrivals > 0 else 0
            print(f"P({n}) = {prob:.4f}")

        plt.bar(range(11), [x / total_arrivals for x in self.pn_counter])
        plt.xlabel("n (packets in system)")
        plt.ylabel("P(n)")
        plt.title("Probability that arriving packet sees n packets")
        plt.grid()
        plt.show()

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python lab1_template.py <npkts> <lambda>")
        sys.exit(1)

    npkts = int(sys.argv[1])
    lambd = float(sys.argv[2])
    sim = Simulator(npkts, lambd)
    sim.run()
