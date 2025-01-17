## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
## Description

    Submission for NUS Maritime Hackathon 2025.
    Team name: hello world.
---
## Features

- Given a CSV file with missing "annotation_severity" columns, generate_severity.py will predict the severity based on the description and factors from other columns and append the severity prediction onto the end of each row (0 & 1 - Low, 2 - Medium, 3 - High)
---
## Installation

### Initial Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Simplificatedd/MaritimeHackathon2025
   ```
2. Go into the directory:
    ```bash
    cd MaritimeHackathon2025
    ```
3. Set Up a Virtual Environment (Optional but Recommended)
    ```bash
    python -m venv venv
    ```
    or
    ```
    python3 -m venv
    ```
    Then activate Virtual Environment:
    ```
    source venv/bin/activate
    ```
    or on Windows:
    ```
    venv\Scripts\activate
    ```
4. Install Dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage
1. Place CSV file with missing "annotation_severity" column into the MaritimeHackathon2025 directory (main directory).

2. Rename the CSV file to 'input.csv'

3. Run the application:
    ```
    python generate_severity.py
    ```
    or
    ```
    python3 generate_severity.py
    ```

4. Output file will be saved to output.csv
---
## Contributors
- **Tang Ki Wang, Raven**
- **Pendurkar Aditi Nishant**
- **Isaac Lim Cheng Ng**
- **Ng Zheng An**