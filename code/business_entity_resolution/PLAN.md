# PLAN: Amazon ML Challenge 2026 (read this first)

> **This file is instructions for you to READ.** You don't run or open it as a program. The only
> thing you "run" is one command in Step 4 below.

---

## 0. Which file is what (nothing needs renaming)

You receive **one file**:

| File you receive | What it is | What to do with it |
|---|---|---|
| **`SEND_TO_MAC_AmazonML_code_and_data.zip`** (about 1.1 GB) | Everything: the program **and** Amazon's data | Put it on the Mac and double-click it to unzip |

*(There's also `CODE_ONLY_AmazonML_no_data.zip`, about 80 KB. It's the same program without the data.
Ignore it unless someone asks for it.)*

Unzipping gives you a folder called **`AmazonML_Project`**. Inside it:

| Inside `AmazonML_Project` | What it is | Do you touch it? |
|---|---|---|
| `START_HERE.txt` | The short version of these steps | Read it |
| `PLAN.md` | This guide | Read it |
| `run.sh` | **The program starter.** Double-clicking it does nothing; start it by typing `bash ~/AmazonML_Project/run.sh` in Terminal (Step 4) | Run it (Step 4) |
| `RUN_ME.command` | The same starter, for double-clicking (macOS may block it; Step 4 is more reliable) | Optional |
| `OFFICIAL_DATA_student_resource.zip` | Amazon's data, found automatically | **Don't open, move or rename it** |
| `team_config.json` | Where you type the team name | Edit it (Step 3) |
| `src/`, `tests/`, `configs/`, `docs/` | The program's code and documentation | Don't touch |
| `README.md`, `AGENT_BRIEF.md` | Technical notes and AI-assistant instructions | Only if curious |

**Don't rename anything.** The names above are already final.

Folders that **appear by themselves** while it runs:

| New folder | What it is |
|---|---|
| `final/` | ⭐ **The files to upload.** `SUBMIT_THESE.txt` inside names them |
| `logs/` | A diary of what happened, for troubleshooting |
| `runs/`, `cache/`, `data/`, `experiments/`, `.venv/` | The program's working space. Leave them alone |

---

## 1. What is this project? (30-second version)

Amazon gave us **3 lists of businesses**. The same shop appears in several lists, written
differently:

| List 1 | List 2 | List 3 |
|---|---|---|
| Ram Marketing Pvt Ltd | राम मार्केटिंग प्राइवेट लिमिटेड (Hindi) | Ram Marketing Private Limited |

The program finds which records are the same business. Amazon scores how accurate it is. Wrong
matches cost more points than missed ones.

---

## 2. What is already done

- ✅ **Program:** written and tested. Amazon's own format checker says **PASS**.
- ✅ **Data:** Amazon's data is inside the zip.
- ❌ **Not done yet:** the big run (that's what the Mac does) and uploading to Unstop.

---

## 3. Steps on the MacBook (about 10 minutes of your time)

### Step 1: Put the zip in your Home folder
1. Copy `SEND_TO_MAC_AmazonML_code_and_data.zip` to the Mac, by USB, AirDrop or Google Drive.
2. In Finder, click **Go** in the top menu bar, then **Home** (the folder with your name).
3. Move the zip into that Home folder.
4. ⚠️ Not the Desktop or Documents.

### Step 2: Unzip it
Double-click the zip. A folder called **`AmazonML_Project`** appears. You can delete the zip
afterwards.

### Step 3: Type your team name
1. Open `AmazonML_Project` and right-click **`team_config.json`**.
2. Choose **Open With > TextEdit**.
3. Change it to look like this, keeping the quotes:
   ```json
   {
     "team_name": "Our Team Name",
     "team_members": "Name 1, Name 2, Name 3"
   }
   ```
4. Save (Cmd + S) and close.

### Step 4: Start the program (the only "run" step)

> ⚠️ **Double-clicking `run.sh` does nothing on a Mac. That's normal.** It only works when you
> type a command in the **Terminal** app, like this:

1. Press **Cmd + Space**. A search bar appears.
2. Type **Terminal** and press **Enter**. A window with a blinking cursor opens.
3. Click inside that window and type exactly this (or copy-paste it). There's one space after
   `bash`, and `~` means "my Home folder":
   ```bash
   bash ~/AmazonML_Project/run.sh
   ```
4. Press **Enter**.

**What you should see within a few seconds:**
```text
==================================================================
 AMAZON ML CHALLENGE - STARTED (auto)   ...
 Keep this Terminal window OPEN and the Mac plugged in (lid open).
 Text will keep appearing below - that is the progress.
==================================================================
>>> STEP 1: preparing Python (first time: 2-5 minutes, needs internet; later: seconds)
```
After that, many more lines appear ("Collecting numpy…", then lines starting with a date and
time). **That's the progress. You don't type anything else.**

- **"No such file or directory":** the folder isn't directly in your Home folder. Either move it
  there (Step 1), or type `bash ` (with a space), drag the file **`run.sh`** from Finder into the
  Terminal window, and press Enter.
- **Download errors:** connect the Mac to Wi-Fi, then repeat Step 4.

### Step 5: Leave it running
- 🔌 Keep the Mac **plugged in** with the **lid open**.
- Don't close the Terminal window.
- If it stops for any reason, repeat **Step 4**. It continues where it stopped.
- It's finished when Terminal says **`finished OK`**.

---

## 4. What happens after Step 4 (automatic; estimated times)

| # | What the program does | About how long |
|---|---|---|
| 1 | Installs its own Python (no password needed) | 2–5 min |
| 2 | Quick test on a small sample | 3–5 min |
| 3 | Full run on all the data | about 1.5–2 hours |
| 4 | ✅ **First upload files appear in `final/`** | – |
| 5 | Tries about 11 improvements and records each score | several hours |
| 6 | Picks the best, re-checks it, and updates `final/` | – |

---

## 5. Uploading (team leader only)

1. Open **`AmazonML_Project/final/SUBMIT_THESE.txt`**. It names the two files.
2. On **Unstop** (logged in as the **team leader**), open the ML Challenge round.
3. **Leaderboard:** upload `final/matching_results.tsv`. You should see a score.
4. **Final submission:** upload `final/<TeamName>_submission.zip`.
5. Tick any declarations yourself.

💡 **Tip:** upload `matching_results.tsv` once as soon as `final/` appears (after about 2 hours),
then upload the final best files again before the deadline.

**Deadline:** 📅 **27 September 2026, 11:59 PM IST** (please confirm on Unstop). Aim to finish
uploads by **about 8 PM IST**.

---

## 6. Using an AI assistant on the Mac (optional)

Open the assistant (for example Claude Code) inside the `AmazonML_Project` folder and say:

> **"Read AGENT_BRIEF.md and follow it."**

It then keeps improving the program by itself. You still do the uploads.

---

## 7. If something goes wrong

| What you see | What to do |
|---|---|
| "could not verify RUN_ME.command" | Ignore it and use Step 4 (the Terminal way) |
| Double-clicked `run.sh` and nothing happened | Normal. Use Terminal: `bash ~/AmazonML_Project/run.sh` (Step 4) |
| "No such file or directory" | The folder isn't directly in Home (Step 1), or type `bash ` and drag `run.sh` into Terminal |
| Terminal seems stuck on "STEP 1" | It's installing (2–5 min the first time). Wait. It needs internet |
| "Another run is already active" | It's already running in another Terminal window. Wait for it |
| "Could not find the official data" | Make sure `OFFICIAL_DATA_student_resource.zip` is still inside `AmazonML_Project` and wasn't opened or renamed |
| It stopped or crashed | Run Step 4 again; finished parts are skipped |
| Anything else | Send the newest file from `AmazonML_Project/logs/` to the team |
