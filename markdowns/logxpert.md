# logXpert

## My Role
As the initiator of the LogXpert project, I contributed to the technical direction.  This project began with an idea to improve log analysis, and I took on a technical lead role to help get it started.  The design of LogXpert follows an approach informed by my research into LLMs and log analysis techniques.  The algorithms are based on this research and aim to address key challenges in the field.  I also developed key code components, including parts for model fine-tuning and data processing.  The progress of LogXpert is really a result of teamwork, and I'm glad to have played a part in the technical development.
## What is LogXpert About?
**Project Introduction: LogXpert - Intelligent Log Analysis Tool Powered by LLM**

**1. Project Overview**

*   **LogXpert** is an innovative log analysis tool driven by a Large Language Model (LLM), developed by the SBC R&D Team. 
*   Its primary goal is to **enhance** the efficiency and intelligence of log analysis, and to **reduce** the reliance on expert resources during troubleshooting.

**2. Background and Challenges**

*   In the SBC R&D domain, analyzing vast amounts of logs is a **time-consuming and stressful** task, especially when dealing with sudden and critical incidents.
*   **LogXpert's objective** is to leverage the power of LLMs to achieve a faster, smarter, and more efficient log analysis process, **minimizing** manual intervention from engineers in troubleshooting.

**3. Core Functionality and User Workflow**

*   **Engineer User Workflow:**
    *   Users can conveniently upload logs via a **web interface**.
    *   The system **automatically constructs a prompt** and sends it to the LogXpert system.
    *   LogXpert returns **root cause analysis and solutions within seconds**.
    *   Underlying Technology: Based on a **fine-tuned model** on NVIDIA A100 GPU, employing the **vLLM inference framework**.

*   **AI Expert User Workflow:**
    *   Continuous Model Optimization: Performance is continuously improved through **Supervised Fine-Tuning (SFT)** (engineer data) and **Direct Preference Optimization (DPO)** (user feedback).
    *   **NMLP Platform** is utilized for model fine-tuning and validation.

**4. Demo and Use Cases**

*   The project will demonstrate LogXpert's capabilities through **practical use case demos**:
    *   **Use Case A:** Quickly gain insights from log snippets.
    *   **Use Case B:** Conduct in-depth analysis of complete log files.
    *   An **interactive web interface** is provided for log analysis operations.

**5. Data and Model Highlights**

*   **Rigorous Data Curation Pipeline:** Includes steps for data collection, labeling, preprocessing, and dataset preparation to ensure data quality.
*   **High-Quality Dataset:** Built by an experienced team of engineers, emphasizing data quality, diversity, and quantity (over 1000 entries for SFT, 500+ for DPO).
*   **Optimized Model Solution:**  Chooses the **fine-tuned LLama 2 70B model** to leverage its capabilities in complex problem solving; employs **QLoRA technology** for efficient fine-tuning under limited resources.

**6. Proof of Concept and Future Outlook**

*   **Significant Proof of Concept Results:** LogXpert demonstrates significantly better performance than baseline LLM models in root cause identification for SBC.
*   **Continuous Improvement Directions:** Committed to addressing the challenges of large log file analysis, exploring larger context windows, and more efficient models (such as Llama 3).

## Awards
* **CNS Full Stack Award 2023** - Jan 2024
* **CNS Hackathon 2023 - Power of Gen** - 1st Place Winner, Oct 2023
* **CNS Quality Week** - 4th Place Winner, Nov 2023

## Invited Presentations
### LogXpert Project Presentations & Engagements

* **CNS P&E Knowledge Transfer Session** -  Co-presenter (1 of 2) to a large audience of 600+ Participants (Feb '25)
* **CNS Leader's Team Meeting - LogXpert Sharing** - Presented by SBC R&D Director to 30+ Participants (Nov '24)
* **ICT Girls Sharing at Ocean University of China** - Co-presenter (1 of 4) to 40+ Students from Ocean University of China (Apr '24)
* **NI Services Mini Workshop** - Co-presenter (1 of 2) to 80+ Participants (Apr '24)
* **Bell Labs' MLA&A Workshop** - Co-presenter (1 of 2) to a large audience of 300+ Participants (Mar '24)
* **CFX “it's myday” Activity - LogXpert Sharing** - Sole Presenter to 100+ Participants (Dec '23)
* **NSB Expert Club's 2nd LLM Workshop - LogXpert Sharing** - Sole Presenter to 100+ Participants (Dec '23)
* **SBC Leaders' Team - LogXpert Sharing** - Sole Presenter to 10+ Participants (Nov '23)
* **SBC Friday Sharing Session - LogXpert Sharing** - Co-presenter (1 of 2) to a large audience of 100+ Participants (Nov '23)