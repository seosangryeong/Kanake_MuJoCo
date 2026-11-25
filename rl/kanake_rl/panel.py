"""Small Tk control window for play_ui.py: terrain, level, driving mode, speeds and policy picker.

Modelled on Teraffe's ``loco_panel.py``. The simulation loop calls ``pump()`` every step
(non-blocking Tk update) and reads requests with ``take_requests()``; nothing here touches the
simulation directly.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

TERRAIN_LABELS = {
    "flat": "평지", "wave": "물결", "stairs": "계단(높이맵)", "ledge": "턱", "slope": "경사",
    "rough": "요철", "bumps": "둔덕",
}


class ControlPanel:
    def __init__(self, kinds, levels, state, limits, policies=None, current_policy=None):
        self.limits = limits  # speed=max m/s, head=max rad
        self.requests = []
        self.closed = False
        self.root = tk.Tk()
        self.root.title("Kanake 제어판")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        pad = dict(padx=6, pady=3)

        # ---- terrain
        box = ttk.LabelFrame(self.root, text="지형")
        box.grid(row=0, column=0, sticky="nsew", **pad)
        self.kind = tk.StringVar(value=state["kind"])
        for i, kind in enumerate(kinds):
            ttk.Radiobutton(box, text=TERRAIN_LABELS.get(kind, kind), value=kind, variable=self.kind).grid(
                row=i % 4, column=i // 4, sticky="w", padx=4)
        ttk.Label(box, text=f"레벨 (0 쉬움 ~ {levels - 1} Teraffe 최종 난이도)").grid(row=4, column=0, columnspan=2,
                                                                              sticky="w", padx=4)
        self.level = tk.IntVar(value=state["level"])
        ttk.Spinbox(box, from_=0, to=levels - 1, textvariable=self.level, width=4).grid(row=4, column=2, sticky="w")
        ttk.Button(box, text="지형 적용 / 다시 시작 (같은 종류도 새로 생성)", command=self._terrain).grid(
            row=5, column=0, columnspan=3, sticky="ew", padx=4, pady=4)

        # ---- driving
        drive = ttk.LabelFrame(self.root, text="운전")
        drive.grid(row=1, column=0, sticky="nsew", **pad)
        self.mode = tk.StringVar(value="keyboard")
        ttk.Radiobutton(drive, text="키보드 (뷰어 창에서 키를 누르는 동안: ↑ 가속 ↓ 브레이크 ←/→ 회전, Shift+←/→ 시선)",
                        value="keyboard", variable=self.mode, command=self._mode).grid(
            row=0, column=0, columnspan=6, sticky="w")
        ttk.Radiobutton(drive, text="자동 (아래 고정 명령으로 계속 주행)", value="auto",
                        variable=self.mode, command=self._mode).grid(row=1, column=0, columnspan=6, sticky="w")

        def field(r, c, label, value, unit):
            ttk.Label(drive, text=label).grid(row=r, column=c, sticky="e", padx=2)
            var = tk.StringVar(value=f"{value:g}")
            ttk.Entry(drive, textvariable=var, width=6).grid(row=r, column=c + 1, sticky="w")
            ttk.Label(drive, text=unit).grid(row=r, column=c + 2, sticky="w")
            return var

        ttk.Label(drive, text="키보드").grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.k_speed = field(3, 0, "최고 속도", limits["speed"], "m/s")
        self.k_ramp = field(4, 0, "가속 시간", 1.0, "s")
        self.k_turn = field(5, 0, "회전 속도", 0.6, "rad/s")
        ttk.Label(drive, text="자동 명령").grid(row=2, column=3, columnspan=3, sticky="w", pady=(6, 0))
        self.a_speed = field(3, 3, "속도", round(limits["speed"] * 0.66, 3), "m/s")
        self.a_dir = field(4, 3, "방향", 0.0, "° (몸 기준)")
        self.a_head = field(5, 3, "머리", 0.0, "°")
        ttk.Button(drive, text="속도·명령 적용", command=self._speeds).grid(row=6, column=0, columnspan=6, sticky="ew",
                                                                     padx=4, pady=4)
        ttk.Button(drive, text="■ 급정지 (명령을 즉시 0으로)", command=lambda: self.requests.append(("stop",))).grid(
            row=7, column=0, columnspan=6, sticky="w", padx=4)
        ttk.Label(drive, text=(f"학습 범위: 속도 0~{limits['speed']:g} m/s, 머리 ±{math.degrees(limits['head']):.0f}°"
                               + (" (이 정책은 머리 명령 없음)" if limits["head"] == 0 else "")
                               + ". 자동 방향은 적용 순간의 몸 방향 기준이며, 몸이 따라 돌면서 유지됩니다."),
                  foreground="#666", wraplength=520, justify="left").grid(row=8, column=0, columnspan=6, sticky="w",
                                                                           padx=4)

        # ---- physics
        phys = ttk.LabelFrame(self.root, text="마찰 (실행 중 바로 적용, 정책을 바꿔도 유지)")
        phys.grid(row=2, column=0, sticky="nsew", **pad)
        self.f_axial = tk.StringVar(value=f"{state['friction'][0]:g}")
        self.f_lateral = tk.StringVar(value=f"{state['friction'][1]:g}")
        for col, (label, var) in enumerate((("링크 축 방향(앞뒤)", self.f_axial), ("링크 옆 방향", self.f_lateral))):
            ttk.Label(phys, text=label).grid(row=0, column=2 * col, sticky="e", padx=2)
            ttk.Entry(phys, textvariable=var, width=6).grid(row=0, column=2 * col + 1, sticky="w", padx=(0, 8))
        ttk.Button(phys, text="마찰 적용", command=self._friction).grid(row=0, column=4, padx=4)
        ttk.Label(phys, text="학습 기본값: 축 0.2 / 옆 1.0 (학습 중 랜덤화 0.14~0.28 / 0.8~1.2). 옆 > 축이어야 serpentine으로 전진",
                  foreground="#666", wraplength=520, justify="left").grid(row=1, column=0, columnspan=5, sticky="w",
                                                                           padx=4)

        # ---- view
        view = ttk.LabelFrame(self.root, text="보기")
        view.grid(row=3, column=0, sticky="nsew", **pad)
        self.collision = tk.StringVar(value="off")
        for value, label in (("off", "CAD 메쉬"), ("only", "충돌 캡슐"), ("both", "둘 다")):
            ttk.Radiobutton(view, text=label, value=value, variable=self.collision,
                            command=lambda: self.requests.append(("view", self.collision.get()))).pack(
                side="left", padx=4)
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(view, text="카메라 따라가기", variable=self.follow,
                        command=lambda: self.requests.append(("follow", bool(self.follow.get())))).pack(
            side="left", padx=8)

        # ---- status
        stat = ttk.LabelFrame(self.root, text="현재 상태")
        stat.grid(row=4, column=0, sticky="nsew", **pad)
        self.status = tk.StringVar(value="")
        ttk.Label(stat, textvariable=self.status, font=("TkFixedFont", 10), justify="left").pack(anchor="w", padx=4)
        self.message = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.message, foreground="#b04000", wraplength=520).grid(
            row=5, column=0, sticky="w", padx=8)

        # ---- policies
        self.policies = list(policies or [])
        if self.policies:
            pol = ttk.LabelFrame(self.root, text="정책 (실험별 가장 많이 학습된 체크포인트) — 더블클릭 또는 버튼으로 불러오기")
            pol.grid(row=0, column=1, rowspan=6, sticky="nsew", **pad)
            self.root.columnconfigure(1, weight=1)
            tree = ttk.Treeview(pol, columns=("iter", "desc"), height=16, selectmode="browse")
            tree.heading("#0", text="정책")
            tree.heading("iter", text="학습 횟수")
            tree.heading("desc", text="설명")
            tree.column("#0", width=150, stretch=False)
            tree.column("iter", width=90, stretch=False, anchor="e")
            tree.column("desc", width=480)
            tree.grid(row=0, column=0, sticky="nsew")
            pol.rowconfigure(0, weight=1)
            pol.columnconfigure(0, weight=1)
            tree.tag_configure("current", background="#dbe9ff")
            for i, p in enumerate(self.policies):
                it = f"{p['iteration']}" + ("" if p.get("finished", True) else " (학습 중)")
                tree.insert("", "end", iid=str(i), text=p["key"], values=(it, p["description"]))
            tree.bind("<Double-1>", lambda e: self._policy())
            self.tree = tree
            self.policy_text = tk.StringVar(value=f"현재: {current_policy}")
            ttk.Label(pol, textvariable=self.policy_text, foreground="#666", wraplength=700, justify="left").grid(
                row=1, column=0, sticky="w", padx=4)
            ttk.Button(pol, text="선택한 정책 불러오기", command=self._policy).grid(row=2, column=0, sticky="ew",
                                                                          padx=4, pady=4)
            self.set_current(current_policy)

    # ---- callbacks
    def _close(self):
        self.closed = True
        self.root.destroy()

    def _terrain(self):
        self.requests.append(("terrain", self.kind.get(), int(self.level.get())))

    def _friction(self):
        try:
            axial, lateral = float(self.f_axial.get()), float(self.f_lateral.get())
        except ValueError:
            self.message.set("마찰: 숫자를 입력하세요")
            return
        axial, lateral = min(max(axial, 0.01), 3.0), min(max(lateral, 0.01), 3.0)
        self.f_axial.set(f"{axial:g}")
        self.f_lateral.set(f"{lateral:g}")
        note = "" if lateral > axial else " — 옆 ≤ 축이면 serpentine 추진력이 없습니다"
        self.message.set(f"마찰 적용: 축 {axial:g} / 옆 {lateral:g}{note}")
        self.requests.append(("friction", axial, lateral))

    def _mode(self):
        self.requests.append(("mode", self.mode.get()))

    def _policy(self):
        selection = self.tree.selection()
        if not selection:
            self.message.set("정책을 먼저 선택하세요")
            return
        policy = self.policies[int(selection[0])]
        self.message.set(f"{policy['key']} 불러오는 중... (관측 구조가 다르면 뷰어가 다시 열립니다)")
        self.requests.append(("policy", str(policy["checkpoint"])))

    def _speeds(self):
        try:
            values = [float(v.get()) for v in (self.k_speed, self.k_ramp, self.k_turn,
                                               self.a_speed, self.a_dir, self.a_head)]
        except ValueError:
            self.message.set("숫자를 입력하세요")
            return
        top, head = self.limits["speed"], math.degrees(self.limits["head"])
        values[0] = min(max(values[0], 0.0), top)
        values[1] = min(max(values[1], 0.05), 5.0)
        values[2] = min(max(values[2], 0.0), 1.5)
        values[3] = min(max(values[3], 0.0), top)
        values[5] = min(max(values[5], -head), head)
        for var, value in zip((self.k_speed, self.k_ramp, self.k_turn, self.a_speed, self.a_dir, self.a_head), values):
            var.set(f"{value:g}")  # Show the clamped values that are actually used.
        self.message.set("적용됨 (학습 범위 밖 값은 범위 안으로 잘림)")
        self.requests.append(("speeds", *values))

    # ---- loop interface
    def set_current(self, checkpoint):
        if not self.policies:
            return
        self.policy_text.set(f"현재: {checkpoint}")
        for i, p in enumerate(self.policies):
            current = str(p["checkpoint"]) == str(checkpoint)
            self.tree.item(str(i), tags=("current",) if current else ())
            if current:
                self.tree.selection_set(str(i))
                self.tree.see(str(i))

    def take_requests(self):
        out, self.requests = self.requests, []
        return out

    def set_status(self, text):
        if not self.closed:
            self.status.set(text)

    def pump(self):
        if self.closed:
            return
        try:
            self.root.update()
        except tk.TclError:
            self.closed = True
