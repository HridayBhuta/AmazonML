# PLAN: Amazon ML Challenge 2026 (read this first)

Hi! This explains what this project is, what's already done, and exactly what you need to do.
No coding needed.

---

## 1. What is this project?

Amazon gave us **3 lists of businesses**. The same shop can appear in more than one list, written
differently:

| List 1 | List 2 | List 3 |
|---|---|---|
| Ram Marketing Pvt Ltd | राम मार्केटिंग प्राइवेट लिमिटेड (Hindi script) | Ram Marketing Private Limited |
| 105 Elm Street, Morganton, NC | 105 ELM ST, MORGANTON, NC | Morganton, NC, 105 Elm St |

**Our job:** for every business in List 1, find its copies in Lists 2 and 3, or say "no copies".
A program does this automatically, and Amazon scores how accurate it is.

**Scoring:** a wrong match costs more points than a missed match, so the program only says "match"
when it's confident.

---

## 2. What is already done

- ✅ **Data:** the official data is downloaded and verified. It's the 1.09 GB zip inside the package.
- ✅ **Program:** written and tested. The whole chain works, and Amazon's own checker says **PASS**.
- ✅ **Mistakes fixed:** the code was reviewed and 27 problems were fixed.
- ✅ **Package:** everything is packed into one zip for a MacBook.
- ❌ **Not done yet:** the full-size run (it needs a stronger computer, which is why we use your
  MacBook) and the uploads to the competition website.

---

## 3. What YOU do on the MacBook (about 10 minutes of your time)

### Step 1: Copy the zip to the Mac
Get the file **`amazon_ml_er_mac_WITH_DATA.zip`** (1.09 GB) onto the Mac by USB drive, AirDrop or
Google Drive.

### Step 2: Unzip it in your Home folder
1. In Finder, click **Go** in the top menu, then **Home**.
2. Move the zip there and double-click it. You get a folder called **`amazon_ml_er`**.
3. ⚠️ Don't use the Desktop or Documents: iCloud may try to upload many gigabytes.

### Step 3: Add your team details (optional but recommended)
1. Open `amazon_ml_er/team_config.json` with TextEdit.
2. Fill it in like this:
   ```json
   {
     "team_name": "Our Team Name",
     "team_members": "Name 1, Name 2, Name 3"
   }
   ```
3. Save it.

### Step 4: Start it
1. Open the **Terminal** app (press Cmd + Space, type `Terminal`, press Enter).
2. Type `cd ` (the letters c and d, then a space). Don't press Enter yet.
3. Drag the **`amazon_ml_er`** folder from Finder into the Terminal window, then press **Enter**.
4. Type this and press **Enter**:
   ```bash
   bash run.sh
   ```

### Step 5: Leave it alone
- 🔌 Keep the Mac **plugged in** with the **lid open**.
- You can keep using the Mac for light things.
- The text scrolling in Terminal is normal; that's the progress.
- If it stops for any reason (crash, lid closed, power), just repeat Step 4. It continues from where
  it stopped.

---

## 4. What happens after you press Enter (automatic)

| # | What the computer does | About how long (estimate) |
|---|---|---|
| 1 | Installs its own Python (no password needed) | 2–5 min |
| 2 | Quick test on a small sample, to make sure everything works | 3–5 min |
| 3 | Full real run: cleans 22 million rows, builds shortlists, trains the model, makes answers | about 1.5–2 hours |
| 4 | ✅ **First submission files ready** in the `final/` folder | seconds |
| 5 | Tries about 11 variations to improve the score and records each result | several hours |
| 6 | Picks the best version, checks it again, and replaces the files in `final/` | seconds |

The first valid submission exists after about 2 hours, even if the improvements take longer.

When it's completely finished, Terminal shows **"finished OK"**.

---

## 5. Uploading (team leader only)

1. Open **`amazon_ml_er/final/SUBMIT_THESE.txt`**. It names the exact two files.
2. On **Unstop** (logged in as the **team leader**), open the ML Challenge round.
3. Upload **`final/matching_results.tsv`** to the **leaderboard**. You should see a score
   (status "SCORED").
4. Upload **`final/<TeamName>_submission.zip`** as the **final submission package**.
5. Tick any declarations Unstop asks for yourself.

💡 **Tip:** upload once as soon as the first files appear (after about 2 hours). Upload again with
the final best files before the deadline.

---

## 6. Deadline

📅 **27 September 2026, 11:59 PM IST** (from the schedule we copied; please check it on Unstop).

**Suggested timeline:**
- **Start the run** as early as possible.
- **About 2 hours after starting:** upload the first leaderboard file.
- **By 27 Sep, about 8 PM IST:** stop experimenting, upload the final leaderboard file and the final
  zip, and check they show as received.

---

## 7. Using an AI assistant on the Mac (optional, for better scores)

If you use an AI coding assistant (like Claude Code) on the Mac, open it in the `amazon_ml_er`
folder and say:

> **"Read AGENT_BRIEF.md and follow it."**

That file lets the AI keep improving the program by itself: it tries ideas, records every result,
compares them, and packs the best one. It also lists the competition rules the AI must never break.
You still do the uploads yourself (Section 5).

---

## 8. If something goes wrong

| Problem | What to do |
|---|---|
| "could not verify RUN_ME.command" | Ignore that file. Use the Terminal method in Step 4. |
| "Another run is already active" | One is still running. Wait, or close all Terminal windows and try again. |
| It stopped or crashed | Run `bash run.sh` again; finished steps are skipped. |
| "Could not find the official data" | Make sure `6ab10eb3b23ba_student_resource.zip` is inside the `amazon_ml_er` folder or in Downloads. |
| Anything else | Send the newest file from the `amazon_ml_er/logs/` folder to the team. |

---

## 9. Where things are (inside `amazon_ml_er/`)

| Folder or file | What it is |
|---|---|
| `final/` | ⭐ The files to upload |
| `final/SUBMIT_THESE.txt` | Exactly which files to upload |
| `experiments/results.tsv` | Score table of every attempt |
| `logs/` | What happened, for troubleshooting |
| `README.md` | Technical details |
| `AGENT_BRIEF.md` | Instructions for an AI assistant |
