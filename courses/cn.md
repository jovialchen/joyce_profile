# Computer Network

# [Certificate](https://jovialchen.github.io/joyce_profile/attachment/computerNetwork.pdf)


## My Notes
* **Layering**:
    * Explains the concept of network layering, where each layer offers specific services to the layer above and utilizes services from the layer below.
    * Illustrates how layers communicate sequentially.
* **Encapsulation**:
    * Details how data is encapsulated as it moves through different layers, changing form (e.g., HTTP payload to TCP segment to IP packet to WiFi frame).
    * Discusses Big Endian and Little Endian byte order and their relevance in network data handling.
* **Shannon Limit**:
    * Introduces the Shannon Limit, which defines the theoretical maximum capacity of a communication channel based on bandwidth and signal-to-noise ratio (S/N).
    * Explains the trade-offs between S/N, hardware costs, and signal strength.
* **Modulation**:
    * Covers various modulation techniques for converting bits into analog signals, including Amplitude Shift Keying (ASK), Phase Shift Keying (PSK), I/Q Modulation, and Quadrature Amplitude Modulation (QAM).
    * Provides examples of where these techniques are used in technologies like Ethernet, DSL, cable modems, and WiFi.
* **Bit Errors and Coding Gain**:
    * Discusses the occurrence of bit errors in physical layer transmissions and how redundancy (coding gain) can improve reliability.
    * Illustrates bitrate calculations in 802.15.4 and 802.11n standards.
* **Clock**:
    * Explains asynchronous and synchronous communication.
    * Focuses on how clock information is encoded and recovered in synchronous communication using methods like Manchester Encoding and 4b5b encoding.
* **IP Fragmentation**:
    * Describes the process of dividing IP packets into smaller fragments when they exceed the Maximum Transmission Unit (MTU) of a link layer.
    * Details the fields involved in IP fragmentation and strategies to avoid it.
* **Error Detection**:
    * Briefly introduces Forward Error Correction (FEC) and various coding algorithms used for error correction.
* **CRC (Cyclic Redundancy Check)**:
    * Explains the principles and computation of CRC for error detection at the link layer.
    * Covers polynomial division and its effectiveness in detecting various error types.
* **Hamming Code**:
    * Details the encoding process of Hamming Code, which allows for the detection and correction of single-bit errors by inserting parity bits into data.
    * Provides a step-by-step example of Hamming Code calculation.
* **TCP**:
    * Gives an overview of the Transmission Control Protocol (TCP), including its position in the OSI model and key functions like reliable transmission, flow control, and congestion control.
    * Explains TCP connection establishment (three-way handshake), data transfer, and termination (four-way handshake).
    * Describes the TCP segment format, including fields such as port numbers, sequence numbers, acknowledgment numbers, and flags.
    * Presents the TCP finite state machine (FSM).
    * Discusses TCP flow control mechanisms, including stop-and-wait and sliding window.
    * Explains TCP congestion control principles and implementations.
* **TLS**:
    * Introduces the Message Authentication Code (MAC) used in the Transport Layer Security (TLS) protocol for message authentication.
    * Clarifies that MACs are for detecting tampering, not transmission errors.
* **NAT**:
    * Explains Network Address Translation (NAT), its purpose, and different NAT types (Full Cone, Restricted Cone, Port Restricted Cone, Symmetric NAT).
    * Addresses potential issues like Hairpinning.
* **HTTP**:
    * Covers the basics of the Hypertext Transfer Protocol (HTTP), including the request/response model.
    * Discusses improvements from HTTP/1.0 to HTTP/1.1 (persistent connections, pipelining) and HTTP/2.0 (multiplexing, header compression).
* **4 Layer Model**:
    * Summarizes the functions and characteristics of the four-layer Internet model (Application, Transport, Network, Link).
* **The End-To-End Principle**:
    * Explains the principle that certain network functions are best implemented at the end hosts rather than within the network itself.
* **Network Security**:
    * Outlines various network attack methods at Layer 2 (e.g., ARP spoofing, MAC flooding, DHCP attacks) and Layer 3 (e.g., ICMP redirect, BGP hijacking).