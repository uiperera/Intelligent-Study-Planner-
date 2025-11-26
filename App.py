
"""
Intelligent Study Planner - Single-file app (fixed & complete)

Features:
- Data sanitization to avoid NoneType crashes
- ML/simple predictor for required study hours (uses sklearn if installed)
- Rule-based priority adjustment
- Allocation to calendar slots based on weekly availability
- Clash detection + greedy redistribution
- Reschedule UI to move hours between days
- Reminders for today's blocks (in-app popups)
- Persistence of subjects, availability, and last generated schedule in JSON
"""

import os
import json
import datetime
from collections import OrderedDict
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

# -------------------------
# Data persistence helpers
# -------------------------
DATA_FILE = "study_planner_data.json"

def today_date():
    return datetime.date.today()

def parse_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def load_data():
    if not os.path.exists(DATA_FILE):
        return {"subjects": {}, "availability": {}}
    with open(DATA_FILE, "r") as f:
        try:
            return json.load(f)
        except Exception:
            # Corrupted file fallback
            return {"subjects": {}, "availability": {}}

def save_data(data):
    # Ensure JSON serializable: convert any non-serializable objects if needed
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

def sanitize_data(data):
    """
    Ensure loaded JSON has expected structure and types.
    - subjects: dict of subjects with safe defaults for missing values
    - availability: keys Mon..Sun with numeric values
    - reminders: keep as-is
    """
    if data is None:
        data = {}
    if 'subjects' not in data or not isinstance(data['subjects'], dict):
        data['subjects'] = {}
    # sanitize each subject
    for name, v in list(data['subjects'].items()):
        if not isinstance(v, dict):
            data['subjects'][name] = {}
            v = data['subjects'][name]
        # deadline: must be YYYY-MM-DD string; if missing set to today +7
        dd = v.get('deadline')
        if dd is None:
            v['deadline'] = (today_date() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        else:
            # check parse
            if parse_date(str(dd)) is None:
                v['deadline'] = (today_date() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            else:
                v['deadline'] = str(dd)
        # difficulty: 1-5
        diff = v.get('difficulty')
        try:
            if diff is None:
                v['difficulty'] = 3
            else:
                v['difficulty'] = int(diff)
                if v['difficulty'] < 1 or v['difficulty'] > 5:
                    v['difficulty'] = 3
        except Exception:
            v['difficulty'] = 3
        # required_hours: float >=0
        rh = v.get('required_hours')
        try:
            if rh is None:
                v['required_hours'] = 2.0
            else:
                v['required_hours'] = float(rh)
                if v['required_hours'] < 0:
                    v['required_hours'] = 2.0
        except Exception:
            v['required_hours'] = 2.0
        # score and past_hours: allow None or numeric
        sc = v.get('score')
        if sc is None:
            v['score'] = None
        else:
            try:
                v['score'] = int(sc)
            except Exception:
                v['score'] = None
        ph = v.get('past_hours')
        if ph is None:
            v['past_hours'] = None
        else:
            try:
                v['past_hours'] = float(ph)
            except Exception:
                v['past_hours'] = None

    # availability keys: ensure Mon..Sun exist and are numeric
    if 'availability' not in data or not isinstance(data['availability'], dict):
        data['availability'] = {}
    for wd in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]:
        val = data['availability'].get(wd)
        try:
            if val is None or val == "":
                data['availability'][wd] = 0.0
            else:
                data['availability'][wd] = float(val)
                if data['availability'][wd] < 0:
                    data['availability'][wd] = 0.0
        except Exception:
            data['availability'][wd] = 0.0

    # reminders leave untouched if present
    if 'reminders' not in data:
        data['reminders'] = {}

    return data

# -------------------------
# Predictor
# -------------------------
try:
    from sklearn.linear_model import LinearRegression
    import numpy as np
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

class SimplePredictor:
    def __init__(self):
        if SKLEARN_AVAILABLE:
            X = np.array([
                [5, 40, 2],
                [5, 55, 3],
                [4, 60, 2],
                [3, 70, 2],
                [2, 85, 1],
                [1, 90, 1],
                [4, 50, 4]
            ])
            y = np.array([10, 8, 6, 4, 2, 1.5, 7])
            self.model = LinearRegression()
            try:
                self.model.fit(X, y)
                self.use_model = True
            except Exception:
                self.model = None
                self.use_model = False
        else:
            self.model = None
            self.use_model = False

    def predict(self, difficulty, past_score, past_hours):
        if past_score is None or past_hours is None:
            base = {1:1.5,2:2.5,3:4.0,4:6.0,5:8.0}
            return float(base.get(difficulty, 4.0))
        if self.use_model:
            import numpy as np
            x = np.array([[difficulty, past_score, past_hours]])
            try:
                pred = float(self.model.predict(x)[0])
            except Exception:
                pred = past_hours + 1.0
            return round(max(1.0, pred),2)
        else:
            if past_score < 50: return round(past_hours + 4,2)
            elif past_score < 70: return round(past_hours + 2,2)
            else: return round(max(1.0, past_hours + 0.5),2)

# -------------------------
# Rule-based functions
# -------------------------
def detect_and_adjust_priorities(subjects):
    names = list(subjects.keys())
    names.sort(key=lambda n: subjects[n]['deadline'])
    for i in range(len(names)-1):
        a = names[i]
        b = names[i+1]
        d1 = subjects[a]['deadline']
        d2 = subjects[b]['deadline']
        if abs((d2-d1).days) <= 2:
            if subjects[a]['difficulty'] < subjects[b]['difficulty']:
                subjects[b]['priority'] +=1
            else:
                subjects[a]['priority'] +=1

def build_calendar_slots(availability, start_date, end_date):
    slots = OrderedDict()
    cur = start_date
    while cur <= end_date:
        weekday = cur.strftime("%a")
        hrs = float(availability.get(weekday,0))
        slots[cur] = hrs
        cur += datetime.timedelta(days=1)
    return slots

def allocate_hours_to_schedule(subjects_input, availability, start_date=None):
    if not subjects_input: return OrderedDict(), {}
    if start_date is None: start_date = today_date()
    max_deadline = max(v['deadline'] for v in subjects_input.values())
    slots = build_calendar_slots(availability, start_date, max_deadline)
    remaining = {name: float(data['required_hours']) for name, data in subjects_input.items()}
    schedule = OrderedDict((d,[]) for d in slots.keys())
    subjects_sorted = sorted(subjects_input.items(), key=lambda kv: (kv[1]['deadline'], -kv[1]['priority']))
    for name, meta in subjects_sorted:
        req = remaining[name]
        for day in schedule.keys():
            if day > meta['deadline']: break
            used = sum(entry['hours'] for entry in schedule[day])
            avail = slots[day]-used
            if avail <= 0: continue
            alloc = min(avail, req)
            if alloc>0:
                schedule[day].append({'subject':name,'hours':round(alloc,2)})
                req -= alloc
                remaining[name] = round(req,2)
                if req<=0.001:
                    remaining[name]=0.0
                    break
    # Secondary pass for remaining
    for name, rem in list(remaining.items()):
        if rem>0:
            for day in schedule.keys():
                if day > subjects_input[name]['deadline']: break
                used = sum(entry['hours'] for entry in schedule[day])
                avail = slots[day]-used
                if avail<=0: continue
                alloc = min(avail, rem)
                schedule[day].append({'subject':name,'hours':round(alloc,2)})
                rem -= alloc
                remaining[name] = round(rem,2)
                if rem<=0.001:
                    remaining[name]=0.0
                    break
    return schedule, remaining

# -------------------------
# Clash detection & redistribution
# -------------------------
def detect_and_redistribute_clashes(schedule, availability):
    if not schedule:
        return schedule
    # Build free capacity per day
    free_slots = OrderedDict()
    for d in schedule.keys():
        weekday = d.strftime("%a")
        total_avail = float(availability.get(weekday, 0))
        used = sum(entry['hours'] for entry in schedule[d])
        free_slots[d] = round(total_avail - used, 6)

    # For each day that is overloaded, try to move entries earlier
    for day in list(schedule.keys()):
        weekday = day.strftime("%a")
        total_avail = float(availability.get(weekday, 0))
        used = sum(entry['hours'] for entry in schedule[day])
        overload = round(used - total_avail, 6)
        if overload <= 0.0001:
            continue
        for entry in list(schedule[day]):
            if overload <= 0:
                break
            subject = entry['subject']
            hrs = entry['hours']
            if hrs <= 0.001:
                continue
            move_amount = min(hrs, overload)
            for prev_day in reversed(list(schedule.keys())):
                if prev_day >= day:
                    continue
                prev_weekday = prev_day.strftime("%a")
                prev_total_avail = float(availability.get(prev_weekday,0))
                prev_used = sum(e['hours'] for e in schedule[prev_day])
                prev_free = round(prev_total_avail - prev_used,6)
                if prev_free <= 0.0001:
                    continue
                take = min(prev_free, move_amount)
                if take <= 0:
                    continue
                schedule[prev_day].append({'subject': subject, 'hours': round(take,2)})
                entry['hours'] = round(entry['hours'] - take, 2)
                move_amount = round(move_amount - take, 2)
                overload = round(overload - take, 2)
                free_slots[prev_day] = round(prev_free - take, 6)
                if entry['hours'] <= 0.0001:
                    try:
                        schedule[day].remove(entry)
                    except ValueError:
                        pass
                if overload <= 0:
                    break
    # Cleanup
    for day in list(schedule.keys()):
        newlist = []
        for e in schedule[day]:
            if e.get('hours',0) > 0.001:
                newlist.append({'subject': e['subject'], 'hours': round(e['hours'],2)})
        schedule[day] = newlist
    return schedule

# -------------------------
# GUI
# -------------------------
class StudyPlannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Study Planner")
        self.root.geometry("900x650")
        self.data = sanitize_data(load_data())
        self.predictor = SimplePredictor()
        self.schedule_cache = None

        # If last_generated_schedule exists, try to load it into schedule_cache
        if self.data.get('last_generated_schedule'):
            try:
                s = OrderedDict()
                for k, v in sorted(self.data['last_generated_schedule'].items()):
                    d = parse_date(k)
                    if d:
                        s[d] = v
                if s:
                    self.schedule_cache = s
            except Exception:
                self.schedule_cache = None

        self.tab_control = ttk.Notebook(root)
        self.tab_subjects = ttk.Frame(self.tab_control)
        self.tab_schedule = ttk.Frame(self.tab_control)
        self.tab_availability = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_subjects, text='Subjects')
        self.tab_control.add(self.tab_schedule, text='Schedule')
        self.tab_control.add(self.tab_availability, text='Availability')
        self.tab_control.pack(expand=1, fill="both")

        self.create_subjects_tab()
        self.create_schedule_tab()
        self.create_availability_tab()

        # start periodic reminder checks (every 30 seconds)
        self.root.after(30*1000, self._periodic_check)

    # -------------------------
    # Subjects Tab
    # -------------------------
    def create_subjects_tab(self):
        frame = self.tab_subjects
        ttk.Button(frame, text="View Subjects", command=self.view_subjects).pack(pady=5)
        ttk.Button(frame, text="Add Subject", command=self.add_subject).pack(pady=5)
        ttk.Button(frame, text="Update Performance", command=self.update_performance).pack(pady=5)
        ttk.Button(frame, text="Edit Subject", command=self.edit_subject).pack(pady=5)
        ttk.Button(frame, text="Delete Subject", command=self.delete_subject).pack(pady=5)
        self.subjects_listbox = tk.Listbox(frame, width=140)
        self.subjects_listbox.pack(pady=10, fill="both", expand=True)
        self.view_subjects()

    def view_subjects(self):
        self.subjects_listbox.delete(0, tk.END)
        for name, v in sorted(self.data.get('subjects',{}).items()):
            line = f"{name} | Deadline: {v['deadline']} | Difficulty: {v['difficulty']} | Req hours: {v.get('required_hours','-')} | Score: {v.get('score')} | Past hrs: {v.get('past_hours')}"
            self.subjects_listbox.insert(tk.END, line)

    def add_subject(self):
        name = simpledialog.askstring("Input", "Enter subject name:")
        if not name: return
        if name in self.data['subjects']:
            messagebox.showerror("Error", "Subject already exists!")
            return
        deadline_str = simpledialog.askstring("Input", "Enter deadline (YYYY-MM-DD):")
        dd = parse_date(deadline_str)
        if dd is None:
            messagebox.showerror("Error", "Invalid date!")
            return
        diff = simpledialog.askinteger("Input", "Difficulty (1-5):", minvalue=1, maxvalue=5)
        past_perf = messagebox.askyesno("Past Performance", "Do you have past performance data?")
        score, past_hours = None, None
        if past_perf:
            score = simpledialog.askinteger("Input", "Enter past score (0-100):", minvalue=0, maxvalue=100)
            past_hours = simpledialog.askfloat("Input", "Enter past hours spent:", minvalue=0)
        req_hours = self.predictor.predict(diff, score, past_hours)
        if 'subjects' not in self.data:
            self.data['subjects'] = {}
        self.data['subjects'][name] = {
            "deadline": dd.strftime("%Y-%m-%d"),
            "difficulty": diff,
            "score": score,
            "past_hours": past_hours,
            "required_hours": req_hours
        }
        save_data(self.data)
        messagebox.showinfo("Added", f"{name} added. Predicted hours: {req_hours}")
        self.view_subjects()

    def update_performance(self):
        name = simpledialog.askstring("Input", "Enter subject name to update performance:")
        if not name or name not in self.data['subjects']:
            messagebox.showerror("Error", "Subject not found!")
            return
        score = simpledialog.askinteger("Input", "Enter past score (0-100):", minvalue=0, maxvalue=100)
        past_hours = simpledialog.askfloat("Input", "Enter past hours spent:", minvalue=0)
        diff = int(self.data['subjects'][name]['difficulty'])
        req = self.predictor.predict(diff, score, past_hours)
        self.data['subjects'][name]['score'] = score
        self.data['subjects'][name]['past_hours'] = past_hours
        self.data['subjects'][name]['required_hours'] = req
        save_data(self.data)
        messagebox.showinfo("Updated", f"Updated required hours: {req}")
        self.view_subjects()

    def edit_subject(self):
        name = simpledialog.askstring("Input", "Enter subject name to edit:")
        if not name or name not in self.data['subjects']:
            messagebox.showerror("Error", "Subject not found!")
            return
        subj = self.data['subjects'][name]
        # allow editing deadline and difficulty
        deadline_str = simpledialog.askstring("Edit", f"Enter new deadline (YYYY-MM-DD) or leave blank to keep ({subj['deadline']}):")
        if deadline_str:
            dd = parse_date(deadline_str)
            if dd is None:
                messagebox.showerror("Error", "Invalid date!")
                return
            subj['deadline'] = dd.strftime("%Y-%m-%d")
        diff = simpledialog.askinteger("Edit", f"Difficulty (1-5) current {subj['difficulty']}:", minvalue=1, maxvalue=5)
        if diff:
            subj['difficulty'] = diff
        # optionally update past performance
        if messagebox.askyesno("Performance", "Update past performance data?"):
            score = simpledialog.askinteger("Input", "Enter past score (0-100):", minvalue=0, maxvalue=100)
            past_hours = simpledialog.askfloat("Input", "Enter past hours spent:", minvalue=0)
            subj['score'] = score
            subj['past_hours'] = past_hours
            subj['required_hours'] = self.predictor.predict(subj['difficulty'], score, past_hours)
        save_data(self.data)
        messagebox.showinfo("Edited", f"{name} updated.")
        self.view_subjects()

    def delete_subject(self):
        name = simpledialog.askstring("Input", "Enter subject name to delete:")
        if not name or name not in self.data['subjects']:
            messagebox.showerror("Error", "Subject not found!")
            return
        if not messagebox.askyesno("Confirm", f"Delete subject {name}?"):
            return
        del self.data['subjects'][name]
        save_data(self.data)
        messagebox.showinfo("Deleted", f"{name} deleted.")
        self.view_subjects()

    # -------------------------
    # Schedule Tab
    # -------------------------
    def create_schedule_tab(self):
        frame = self.tab_schedule
        ttk.Button(frame, text="Generate Schedule", command=self.generate_schedule).pack(pady=5)
        ttk.Button(frame, text="View Today's Plan", command=self.show_today_plan).pack(pady=5)
        ttk.Button(frame, text="Reschedule Block", command=self.reschedule_block).pack(pady=5)
        ttk.Button(frame, text="Set Reminder for Today's Blocks", command=self.set_today_reminders).pack(pady=5)
        self.schedule_text = tk.Text(frame, width=120, height=30)
        self.schedule_text.pack(pady=10, fill="both", expand=True)
        if self.schedule_cache:
            self.generate_schedule_display_from_cache()

    def generate_schedule(self):
        if not self.data.get('availability'):
            messagebox.showerror("Error", "Set availability first!")
            return
        if not self.data.get('subjects'):
            messagebox.showerror("Error", "Add subjects first!")
            return
        subjects_input = {}
        for name, v in self.data['subjects'].items():
            dd = parse_date(v['deadline'])
            if dd is None:
                messagebox.showerror("Error", f"Invalid deadline for subject {name}")
                return
            # safe difficulty conversion
            diff_val = v.get('difficulty')
            try:
                if diff_val is None:
                    diff_val = 3
                else:
                    diff_val = int(diff_val)
                    if diff_val < 1 or diff_val > 5:
                        diff_val = 3
            except Exception:
                diff_val = 3
            # safe required_hours
            try:
                req_hours = float(v.get('required_hours', 0))
                if req_hours < 0:
                    req_hours = 0.0
            except Exception:
                req_hours = 0.0
            subjects_input[name] = {
                'deadline': dd,
                'difficulty': int(diff_val),
                'required_hours': float(req_hours),
                'priority': int(diff_val)
            }
        detect_and_adjust_priorities(subjects_input)
        schedule, remaining = allocate_hours_to_schedule(subjects_input, self.data['availability'])
        schedule = detect_and_redistribute_clashes(schedule, self.data['availability'])
        self.schedule_cache = schedule
        # Persist schedule into JSON (convert dates to strings)
        try:
            self.data['last_generated_schedule'] = {d.strftime("%Y-%m-%d"): schedule[d] for d in schedule.keys()}
            save_data(self.data)
        except Exception:
            pass
        self.generate_schedule_display_from_cache()
        if any(v>0.001 for v in remaining.values()):
            self.schedule_text.insert(tk.END,"\n⚠ Some subjects could not be fully scheduled. Consider increasing availability or starting earlier.\n")
        else:
            self.schedule_text.insert(tk.END,"\nAll required hours scheduled.\n")

    def generate_schedule_display_from_cache(self):
        self.schedule_text.delete("1.0", tk.END)
        if not self.schedule_cache:
            self.schedule_text.insert(tk.END, "No schedule available. Generate a schedule first.\n")
            return
        for day, blocks in sorted(self.schedule_cache.items()):
            if not blocks: continue
            try:
                day_str = day.strftime('%Y-%m-%d')
                dow = day.strftime('%a')
            except Exception:
                dobj = parse_date(str(day))
                if dobj:
                    day_str = dobj.strftime('%Y-%m-%d')
                    dow = dobj.strftime('%a')
                else:
                    day_str = str(day)
                    dow = ""
            self.schedule_text.insert(tk.END, f"\n{day_str} ({dow}):\n")
            for b in blocks:
                self.schedule_text.insert(tk.END, f"  - {b['subject']}: {b['hours']} hours\n")

    def show_today_plan(self):
        if not self.schedule_cache:
            messagebox.showinfo("Info","Generate schedule first!")
            return
        today = datetime.date.today()
        blocks = self.schedule_cache.get(today,[])
        if not blocks:
            messagebox.showinfo("Today's Plan","No study blocks scheduled today.")
            return
        msg = "Today's study plan:\n"
        for b in blocks:
            msg += f" - {b['subject']}: {b['hours']} hours\n"
        messagebox.showinfo("Today's Plan", msg)

    def reschedule_block(self):
        if not self.schedule_cache:
            messagebox.showinfo("Info", "Generate schedule first!")
            return
        day_str = simpledialog.askstring("Reschedule", "Enter date of the block to move (YYYY-MM-DD):")
        if not day_str:
            return
        day = parse_date(day_str)
        if day is None:
            messagebox.showerror("Error","Invalid date format.")
            return
        if day not in self.schedule_cache:
            messagebox.showerror("Error", "No blocks on that date.")
            return
        blocks = self.schedule_cache[day]
        if not blocks:
            messagebox.showerror("Error", "No blocks on that date.")
            return
        subj_names = [f"{i+1}. {b['subject']} ({b['hours']} hrs)" for i,b in enumerate(blocks)]
        choice = simpledialog.askinteger("Choose block", "Which block to move?\n" + "\n".join(subj_names), minvalue=1, maxvalue=len(blocks))
        if not choice:
            return
        block = blocks[choice-1]
        move_hours = simpledialog.askfloat("Hours", f"How many hours to move (max {block['hours']}):", minvalue=0.01, maxvalue=block['hours'])
        if not move_hours:
            return
        target_str = simpledialog.askstring("Target", "Enter target date (YYYY-MM-DD) to move to:")
        target = parse_date(target_str)
        if target is None:
            messagebox.showerror("Error","Invalid target date")
            return
        if target not in self.schedule_cache:
            self.schedule_cache[target] = []
        used = sum(e['hours'] for e in self.schedule_cache[target])
        weekday = target.strftime("%a")
        cap = float(self.data['availability'].get(weekday,0))
        free = cap - used
        if free < 0.01:
            messagebox.showerror("Error","No free capacity on target date.")
            return
        allowed = min(move_hours, free)
        block['hours'] = round(block['hours'] - allowed,2)
        if block['hours'] <= 0.001:
            try:
                self.schedule_cache[day].remove(block)
            except ValueError:
                pass
        self.schedule_cache[target].append({'subject': block['subject'], 'hours': round(allowed,2)})
        try:
            self.data['last_generated_schedule'] = {d.strftime("%Y-%m-%d"): self.schedule_cache[d] for d in self.schedule_cache.keys()}
            save_data(self.data)
        except Exception:
            pass
        messagebox.showinfo("Moved", f"Moved {allowed} hours of {block['subject']} to {target.strftime('%Y-%m-%d')}")
        self.generate_schedule_display_from_cache()

    # -------------------------
    # Availability Tab
    # -------------------------
    def create_availability_tab(self):
        frame = self.tab_availability
        ttk.Label(frame, text="Set weekly availability (hours)").pack(pady=5)
        self.av_entries = {}
        for wd in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]:
            row = ttk.Frame(frame)
            row.pack(fill="x", padx=10, pady=2)
            lbl = ttk.Label(row, text=wd, width=6)
            lbl.pack(side="left")
            ent = ttk.Entry(row, width=10)
            ent.pack(side="left")
            self.av_entries[wd] = ent
            if self.data.get('availability') and wd in self.data['availability']:
                ent.insert(0,str(self.data['availability'][wd]))
        ttk.Button(frame, text="Save Availability", command=self.save_availability).pack(pady=10)

    def save_availability(self):
        if 'availability' not in self.data:
            self.data['availability'] = {}
        for wd, ent in self.av_entries.items():
            try:
                h_text = ent.get().strip()
                if h_text == "":
                    h = 0.0
                else:
                    h = float(h_text)
                if h < 0:
                    raise ValueError
                self.data['availability'][wd] = h
            except Exception:
                self.data['availability'][wd] = 0.0
        save_data(self.data)
        messagebox.showinfo("Saved","Availability saved.")

    # -------------------------
    # Reminders
    # -------------------------
    def set_today_reminders(self):
        if not self.schedule_cache:
            messagebox.showinfo("Info","Generate schedule first!")
            return
        today = datetime.date.today()
        blocks = self.schedule_cache.get(today,[])
        if not blocks:
            messagebox.showinfo("Info","No blocks today to set reminders for.")
            return
        mins = simpledialog.askinteger("Reminder", "Notify me how many minutes before each block starts? (enter 0 for immediate reminder)", minvalue=0)
        if mins is None:
            return
        if 'reminders' not in self.data:
            self.data['reminders'] = {}
        self.data['reminders'][today.strftime("%Y-%m-%d")] = {'minutes_before': mins, 'set_at': datetime.datetime.now().isoformat()}
        save_data(self.data)
        messagebox.showinfo("Reminders set", f"Reminders for {today.strftime('%Y-%m-%d')} set ({mins} minutes before).")
        self.check_and_show_reminders()

    def check_and_show_reminders(self):
        if 'reminders' not in self.data:
            return
        today_key = datetime.date.today().strftime("%Y-%m-%d")
        r = self.data['reminders'].get(today_key)
        if not r:
            return
        if not self.schedule_cache:
            return
        blocks = self.schedule_cache.get(datetime.date.today(), [])
        if not blocks:
            return
        mins = int(r.get('minutes_before',0))
        msg = f"Today's study plan ({datetime.date.today().strftime('%Y-%m-%d')}):\n"
        for b in blocks:
            msg += f" - {b['subject']}: {b['hours']} hours\n"
        if mins == 0:
            messagebox.showinfo("Reminder", msg)
        else:
            messagebox.showinfo("Upcoming Reminder", f"In {mins} minutes you'll have study blocks.\n\n" + msg)

    def _periodic_check(self):
        try:
            self.check_and_show_reminders()
        finally:
            self.root.after(30*1000, self._periodic_check)

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = StudyPlannerGUI(root)
    root.mainloop()
