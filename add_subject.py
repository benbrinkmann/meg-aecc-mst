"""
Add subjects to subjects.csv with a form and file browse buttons.

Run from the repository folder:   python3.11 add_subject.py

For each subject and session: type the subject ID, pick the three files with the
Browse buttons, choose the group, and click "Add". Each row is saved to
subjects.csv (SUBJECTS_CSV in config.py) immediately. Rows already entered are
listed at the bottom and can be removed there.
"""
import csv
import os
import re
import sys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    sys.exit("tkinter is not installed for this Python. On Rocky Linux: "
             "sudo dnf install python3.11-tkinter")

import config
from common import REQUIRED_COLUMNS

# IDs become folder names, so allow only letters, digits, underscore and hyphen.
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
FIF_TYPES = [("FIF files", "*.fif"), ("All files", "*")]


# ----------------------------------------------------------------------------
# subjects.csv reading and writing (no GUI code here)
# ----------------------------------------------------------------------------
def read_rows(path):
    """Return the rows of subjects.csv as a list of dicts ([] if the file does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}; fix or rename it first.")
        # Missing values come back as None; store them as empty strings.
        return [{c: (row.get(c) or "").strip() for c in REQUIRED_COLUMNS} for row in reader]


def write_rows(path, rows):
    """Write rows sorted by subject and session. Writes a temporary file first so a
    failure part way through cannot leave a half written subjects.csv."""
    path = Path(path)
    tmp = path.with_suffix(".csv.tmp")
    rows = sorted(rows, key=lambda r: (r["subject"], r["session"]))
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def next_session(rows, subject):
    """Next free session label for a subject: ses01, ses02, ..."""
    numbers = [int(m.group(1)) for r in rows if r["subject"] == subject
               for m in [re.fullmatch(r"ses(\d+)", r["session"])] if m]
    return f"ses{(max(numbers) + 1) if numbers else 1:02d}"


# ----------------------------------------------------------------------------
# The form
# ----------------------------------------------------------------------------
class SubjectForm(tk.Tk):
    def __init__(self, csv_path):
        super().__init__()
        self.title("Add subjects to subjects.csv")
        self.csv_path = Path(csv_path)
        try:
            self.rows = read_rows(self.csv_path)
        except ValueError as err:
            messagebox.showerror("subjects.csv problem", str(err))
            self.destroy()
            raise SystemExit(1)

        # Start browsing in data/ if it exists, then remember the last folder used.
        data_dir = config.PROJECT_DIR / "data"
        self.last_dir = str(data_dir if data_dir.is_dir() else config.PROJECT_DIR)

        # Form values.
        self.subject = tk.StringVar()
        self.session = tk.StringVar()
        self.fs_subject = tk.StringVar()
        self.group = tk.StringVar(value=config.TARGET_GROUP)
        self.raw = tk.StringVar()
        self.er = tk.StringVar()
        self.mri = tk.StringVar()
        self.no_er = tk.BooleanVar(value=False)
        self.no_mri = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value=f"Saving to {self.csv_path}")

        # Session and FreeSurfer name follow the subject ID until edited by hand.
        self.auto_session = True
        self.auto_fs = True
        self.subject.trace_add("write", self._subject_changed)

        self._build()
        self._refresh_table()
        self.subject_entry.focus_set()

    # --- layout -------------------------------------------------------------
    def _build(self):
        pad = dict(padx=6, pady=3)
        form = ttk.Frame(self, padding=10)
        form.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Subject ID").grid(row=0, column=0, sticky="w", **pad)
        self.subject_entry = ttk.Entry(form, textvariable=self.subject, width=20)
        self.subject_entry.grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(form, text="Session").grid(row=1, column=0, sticky="w", **pad)
        ses_entry = ttk.Entry(form, textvariable=self.session, width=20)
        ses_entry.grid(row=1, column=1, sticky="w", **pad)
        ses_entry.bind("<Key>", lambda e: setattr(self, "auto_session", False))

        ttk.Label(form, text="Group").grid(row=2, column=0, sticky="w", **pad)
        groups = list(config.REFERENCE_GROUPS) + [config.TARGET_GROUP]
        ttk.Combobox(form, textvariable=self.group, values=groups, width=18).grid(
            row=2, column=1, sticky="w", **pad)

        ttk.Label(form, text="FreeSurfer name").grid(row=3, column=0, sticky="w", **pad)
        fs_entry = ttk.Entry(form, textvariable=self.fs_subject, width=20)
        fs_entry.grid(row=3, column=1, sticky="w", **pad)
        fs_entry.bind("<Key>", lambda e: setattr(self, "auto_fs", False))

        # File rows: label, path box, Browse button.
        self._file_row(form, 4, "Resting MEG (tSSS .fif)", self.raw, "Select resting state recording")
        self.er_entry, self.er_button = self._file_row(
            form, 5, "Empty room (tSSS .fif)", self.er, "Select empty room recording")
        ttk.Checkbutton(form, text="No empty room recording (use ad hoc noise covariance)",
                        variable=self.no_er, command=self._toggle_er).grid(
            row=6, column=1, sticky="w", **pad)
        self.mri_entry, self.mri_button = self._file_row(
            form, 7, "MRI wrapper (.fif)", self.mri, "Select MRI wrapper file")
        ttk.Checkbutton(form, text="No usable MRI (use fsaverage scaled to the head shape)",
                        variable=self.no_mri, command=self._toggle_mri).grid(
            row=8, column=1, sticky="w", **pad)

        buttons = ttk.Frame(form)
        buttons.grid(row=9, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="Add", command=self._add).pack(side="left", padx=4)
        ttk.Button(buttons, text="Clear form", command=self._clear).pack(side="left", padx=4)
        ttk.Button(buttons, text="Close", command=self.destroy).pack(side="left", padx=4)
        ttk.Label(form, textvariable=self.status, foreground="#1E4620").grid(
            row=10, column=0, columnspan=3, sticky="w", **pad)

        # Table of rows already in subjects.csv.
        table = ttk.Frame(self, padding=(10, 0, 10, 10))
        table.grid(row=1, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        cols = ("subject", "session", "group", "raw_fif", "er_fif", "mri_fif")
        self.tree = ttk.Treeview(table, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (90, 70, 80, 220, 180, 180)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        ttk.Button(table, text="Remove selected row", command=self._remove).grid(
            row=1, column=0, sticky="w", pady=(6, 0))

    def _file_row(self, parent, row, label, var, title):
        """One labelled path box with a Browse button. Returns (entry, button)."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        entry = ttk.Entry(parent, textvariable=var, width=70)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        button = ttk.Button(parent, text="Browse…", command=lambda: self._browse(var, title))
        button.grid(row=row, column=2, padx=6, pady=3)
        return entry, button

    # --- behaviour ----------------------------------------------------------
    def _subject_changed(self, *_):
        subj = self.subject.get().strip()
        if self.auto_session:
            self.session.set(next_session(self.rows, subj) if subj else "")
        if self.auto_fs:
            # Reuse the FreeSurfer name from an earlier session of this subject, if any.
            earlier = [r["fs_subject"] for r in self.rows if r["subject"] == subj]
            self.fs_subject.set(earlier[0] if earlier else subj)

    def _browse(self, var, title):
        path = filedialog.askopenfilename(parent=self, title=title,
                                          initialdir=self.last_dir, filetypes=FIF_TYPES)
        if path:  # empty string or () if the dialog was cancelled
            var.set(path)
            self.last_dir = str(Path(path).parent)

    def _toggle_er(self):
        state = "disabled" if self.no_er.get() else "normal"
        if self.no_er.get():
            self.er.set("")
        self.er_entry.configure(state=state)
        self.er_button.configure(state=state)

    def _toggle_mri(self):
        state = "disabled" if self.no_mri.get() else "normal"
        if self.no_mri.get():
            self.mri.set("")
        self.mri_entry.configure(state=state)
        self.mri_button.configure(state=state)

    def _validate(self):
        """Return a list of problems with the form (empty list if it is OK)."""
        problems = []
        for label, value in (("Subject ID", self.subject.get()), ("Session", self.session.get()),
                             ("FreeSurfer name", self.fs_subject.get()), ("Group", self.group.get())):
            if not ID_PATTERN.match(value.strip()):
                problems.append(f"{label} must be letters, digits, _ or - (no spaces or commas).")
        files = [("Resting MEG", self.raw.get())]
        if not self.no_er.get():
            files.append(("Empty room", self.er.get()))
        if not self.no_mri.get():
            files.append(("MRI wrapper", self.mri.get()))
        for label, value in files:
            if not value.strip():
                problems.append(f"{label}: no file selected.")
            elif not Path(value.strip()).is_file():
                problems.append(f"{label}: file not found: {value}")
            elif "," in value:
                problems.append(f"{label}: path contains a comma, which breaks the CSV.")
        return problems

    def _add(self):
        problems = self._validate()
        if problems:
            messagebox.showerror("Please fix", "\n".join(problems), parent=self)
            return

        new = dict(subject=self.subject.get().strip(), session=self.session.get().strip(),
                   raw_fif=self.raw.get().strip(),
                   er_fif="" if self.no_er.get() else self.er.get().strip(),
                   mri_fif="" if self.no_mri.get() else self.mri.get().strip(),
                   fs_subject=self.fs_subject.get().strip(),
                   group=self.group.get().strip())

        # Same subject and session already listed: confirm before replacing.
        same = [r for r in self.rows
                if r["subject"] == new["subject"] and r["session"] == new["session"]]
        if same and not messagebox.askyesno(
                "Replace?", f"{new['subject']} {new['session']} is already listed. Replace it?",
                parent=self):
            return
        # Same recording used in a different row: probably a mistake.
        reused = [r for r in self.rows if r["raw_fif"] == new["raw_fif"] and r not in same]
        if reused and not messagebox.askyesno(
                "Recording already used",
                f"This recording is already listed for {reused[0]['subject']} "
                f"{reused[0]['session']}. Add it anyway?", parent=self):
            return

        updated = [r for r in self.rows if r not in same] + [new]
        try:
            write_rows(self.csv_path, updated)
        except OSError as err:
            messagebox.showerror("Could not save", str(err), parent=self)
            return
        self.rows = updated
        self._refresh_table()
        self.status.set(f"Saved {new['subject']} {new['session']}. "
                        f"{len(self.rows)} row(s) in {self.csv_path.name}.")
        self._clear(keep_group=True)

    def _remove(self):
        selected = self.tree.selection()
        if not selected:
            return
        keys = {tuple(self.tree.item(i, "values")[:2]) for i in selected}
        if not messagebox.askyesno("Remove", f"Remove {len(keys)} row(s) from subjects.csv?",
                                   parent=self):
            return
        updated = [r for r in self.rows if (r["subject"], r["session"]) not in keys]
        try:
            write_rows(self.csv_path, updated)
        except OSError as err:
            messagebox.showerror("Could not save", str(err), parent=self)
            return
        self.rows = updated
        self._refresh_table()
        self.status.set(f"Removed {len(keys)} row(s).")

    def _clear(self, keep_group=False):
        self.auto_session = self.auto_fs = True
        for var in (self.subject, self.session, self.fs_subject, self.raw, self.er, self.mri):
            var.set("")
        if not keep_group:
            self.group.set(config.TARGET_GROUP)
        self.no_er.set(False)
        self._toggle_er()
        self.no_mri.set(False)
        self._toggle_mri()
        self.subject_entry.focus_set()

    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        for r in sorted(self.rows, key=lambda r: (r["subject"], r["session"])):
            self.tree.insert("", "end", values=(
                r["subject"], r["session"], r["group"], Path(r["raw_fif"]).name,
                Path(r["er_fif"]).name if r["er_fif"] else "(none: ad hoc)",
                Path(r["mri_fif"]).name if r["mri_fif"] else "(none: template)"))


if __name__ == "__main__":
    SubjectForm(config.SUBJECTS_CSV).mainloop()
