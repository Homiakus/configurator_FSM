# fsm_configurator.py
# Полнофункциональный FSM TOML configurator
# Требует: PyQt6, toml, networkx, matplotlib, python-dateutil

import sys, re, toml, traceback, json
from pathlib import Path
from copy import deepcopy
from functools import partial
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QTabWidget,
    QFormLayout, QTextEdit, QMessageBox, QComboBox, QSpinBox, QCheckBox,
    QTreeWidget, QTreeWidgetItem, QSplitter, QGroupBox, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QSize

# Graphing
import networkx as nx
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Utilities & Defaults ---

DEFAULT_SCHEMA_VERSION = "2025-08-13"
DEFAULT_EXAMPLE = {
    "name": "UniversalFSM",
    "description": "Пример FSM",
    "version": "1.0",
    "schema_version": DEFAULT_SCHEMA_VERSION,
    "timezone": "Europe/Berlin",
    "initial_state": "Idle",
    "final_states": ["Final"],
    "variables": [
        {"name": "power_on", "type": "bool", "initial": True, "persist": True, "readonly": False, "description": "Питание"},
        {"name": "counter", "type": "int", "initial": 0},
        {"name": "limit", "type": "int", "initial": 10}
    ],
    "sources": [
        {"id": "uart", "type": "serial", "description": "UART port", "connection": "/dev/ttyUSB0", "baud": 115200},
        {"id": "gpio", "type": "digital_input", "description": "GPIO inputs"},
        {"id": "db", "type": "database", "description": "sqlite", "connection": "sqlite:///data.db"},
        {"id": "internal", "type": "logic", "description": "internal timers/events"}
    ],
    "actions": [
        {"id": "init_idle", "type": "sync", "impl": "module.init_idle"},
        {"id": "start_pump", "type": "async", "impl": "pump.start"},
        {"id": "release_resources", "type": "sync", "impl": "module.release_resources"},
        {"id": "log_and_stop", "type": "sync", "impl": "module.log_and_stop"}
    ],
    "states": [
        {"id": "Idle", "type": "atomic", "on_enter": ["init_idle"], "on_exit": ["release_resources"], "transitions": [
            {"id": "t_uart_and_db", "trigger": "uart.start", "guard": "db:SELECT enabled FROM settings WHERE id=${profile_id}", "target": "Running", "actions": ["start_pump"], "priority": 10, "consume": True},
            {"id": "t_uart_tds_threshold", "trigger": "uart.tds *", "guard": "expr:event.value >= var.limit", "target": "Running", "actions": ["start_pump"]}
        ]},
        {"id": "Running", "type": "compound", "initial_substate": "RunActive", "substates": [
            {"id": "RunActive", "type": "atomic", "transitions": [
                {"id": "t_run_low_pressure", "trigger": "sensor.pressure *", "guard": "expr:event.value < 2.0", "target": "Error", "actions": ["log_and_stop"], "consume": True}
            ], "timeouts": [{"id": "run_max_time", "duration_ms": 600000, "target": "Idle", "actions": ["release_resources"]}]},
            {"id": "RunPaused", "type": "atomic"}
        ], "history": "shallow"},
        {"id": "Error", "type": "atomic", "on_enter": ["log_and_stop"], "transitions": [
            {"id": "t_error_reset", "trigger": "gpio.reset pressed", "guard": "expr:var.power_on == true", "target": "Idle", "actions": []}
        ]},
        {"id": "Sleep", "type": "atomic"}
    ],
    "timers": [
        {"id": "heartbeat", "type": "periodic", "interval_ms": 1000, "event": "internal.heartbeat", "auto_start": True}
    ],
    "metadata": {"author": "denis", "created_at": datetime.utcnow().isoformat() + "Z"}
}

# --- TOML load/save helpers ---
def load_toml_file(path: str) -> Dict[str, Any]:
    try:
        data = toml.load(path)
        return data
    except Exception as e:
        raise

def save_toml_file(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(data, f)

# --- Validators ---
class FSMValidator:
    """
    Набор встроенных валидаторов. Анализирует структуру и возвращает список ошибок/предупреждений.
    """
    def __init__(self, fsm: Dict[str, Any], auto_create_missing_targets: bool = True):
        self.fsm = deepcopy(fsm)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.auto_create = auto_create_missing_targets

    def run_all(self):
        self.errors = []
        self.warnings = []
        try:
            self._check_basic_meta()
            self._check_unique_state_ids()
            self._check_initial_state()
            self._check_states_transitions()
            self._check_sources_in_triggers()
            self._check_variables_placeholders()
            self._check_actions_references()
            self._check_timers()
            self._check_cycles()
        except Exception as e:
            self.errors.append(f"Validator internal error: {e}\n{traceback.format_exc()}")
        return {"errors": self.errors, "warnings": self.warnings}

    def _check_basic_meta(self):
        if "name" not in self.fsm:
            self.warnings.append("Missing top-level 'name' field.")
        if "schema_version" not in self.fsm:
            self.warnings.append("Missing 'schema_version' (recommended).")

    def _collect_state_ids(self) -> set:
        s = set()
        for sdef in self.fsm.get("states", []):
            sid = sdef.get("id")
            if sid:
                s.add(sid)
        return s

    def _check_unique_state_ids(self):
        ids = []
        for s in self.fsm.get("states", []):
            sid = s.get("id")
            if not sid:
                self.errors.append("State with empty id found.")
            else:
                ids.append(sid)
        dup = set([x for x in ids if ids.count(x) > 1])
        if dup:
            self.errors.append(f"Duplicate state ids: {', '.join(sorted(list(dup)))}")

    def _check_initial_state(self):
        initial = self.fsm.get("initial_state")
        if not initial:
            self.errors.append("initial_state not set.")
        else:
            ids = self._collect_state_ids()
            if initial not in ids:
                self.errors.append(f"initial_state '{initial}' not found among states.")

    def _check_states_transitions(self):
        ids = self._collect_state_ids()
        for s in self.fsm.get("states", []):
            for t in s.get("transitions", []):
                target = t.get("target")
                if not target:
                    self.errors.append(f"Transition {t.get('id','<no id>')} in state {s.get('id')} has no target.")
                elif target not in ids:
                    msg = f"Transition target '{target}' (from {s.get('id')}/{t.get('id','<no id>')}) not found among states."
                    if self.auto_create:
                        # auto-create state at bottom
                        ids.add(target)
                        self.fsm.setdefault("states", []).append({"id": target, "type": "atomic"})
                        self.warnings.append(msg + " Auto-created state.")
                    else:
                        self.errors.append(msg)

    def _check_sources_in_triggers(self):
        # triggers like "uart.start" must have 'uart' in sources
        srcs = set([x.get("id") for x in self.fsm.get("sources", []) if x.get("id")])
        for s in self.fsm.get("states", []):
            for t in s.get("transitions", []):
                trig = t.get("trigger","")
                if trig.startswith("regex:") or trig.strip() == "*" or trig.startswith("any:"):
                    continue
                # token before dot or space
                m = re.match(r"^([^.\s:]+)[\.\:]", trig)
                if not m:
                    # allow internal.* and special cases
                    continue
                src = m.group(1)
                if src not in srcs:
                    self.warnings.append(f"Trigger source '{src}' in transition {t.get('id','<no id>')} not declared in sources.")

    def _check_variables_placeholders(self):
        # check db:${var} placeholders and expr: var.* occurrences
        var_names = set([v.get("name") for v in self.fsm.get("variables", []) if v.get("name")])
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
        for s in self.fsm.get("states", []):
            for t in s.get("transitions", []):
                guard = t.get("guard", "")
                for m in pattern.findall(guard):
                    if m not in var_names:
                        self.errors.append(f"Guard in transition {t.get('id','<no id>')} references unknown variable '${{{m}}}'.")

                # expr: usage check
                if guard.startswith("expr:") or guard.startswith("var:"):
                    # look for var.NAME or var.NAME in expr
                    refs = re.findall(r"var\.([A-Za-z_][A-Za-z0-9_]*)", guard)
                    for r in refs:
                        if r not in var_names:
                            self.errors.append(f"Guard expr in transition {t.get('id','<no id>')} references unknown var '{r}'.")

    def _check_actions_references(self):
        actions_ids = set([a.get("id") for a in self.fsm.get("actions", []) if a.get("id")])
        # also support inline actions 'kind:param' — we won't validate those deeply
        for s in self.fsm.get("states", []):
            for t in s.get("transitions", []):
                for a in t.get("actions", []):
                    if isinstance(a, str) and ":" not in a and a not in actions_ids:
                        self.warnings.append(f"Action '{a}' referenced but not found in actions catalog (transition {t.get('id','<no id>')}).")

    def _check_timers(self):
        for t in self.fsm.get("timers", []):
            if t.get("type") == "periodic" and t.get("interval_ms", 0) <= 0:
                self.errors.append(f"Timer {t.get('id')} has invalid interval_ms.")
        for s in self.fsm.get("states", []):
            for to in s.get("timeouts", []):
                if to.get("duration_ms", 0) <= 0:
                    self.errors.append(f"Timeout {to.get('id','<no id>')} in state {s.get('id')} has non-positive duration_ms.")

    def _check_cycles(self):
        # build directed graph from states/transitions and detect simple cycles without timers
        G = nx.DiGraph()
        state_ids = list(self._collect_state_ids())
        G.add_nodes_from(state_ids)
        for s in self.fsm.get("states", []):
            for t in s.get("transitions", []):
                target = t.get("target")
                if target:
                    G.add_edge(s.get("id"), target, transition_id=t.get("id"))
        cycles = list(nx.simple_cycles(G))
        for c in cycles:
            # check if any transition in cycle has a timeout or timer break
            has_timer = False
            for i in range(len(c)):
                a = c[i]
                b = c[(i+1)%len(c)]
                # find transition(s)
                for s in self.fsm.get("states", []):
                    if s.get("id") == a:
                        for t in s.get("transitions", []):
                            if t.get("target") == b:
                                # if this transition has guard like timeout or is triggered by timer? rough heuristic:
                                trig = t.get("trigger","")
                                if trig.startswith("timeout:") or trig.startswith("internal.") or t.get("timeout_ms"):
                                    has_timer = True
            if not has_timer:
                self.warnings.append(f"Cycle detected without timer/timeout: {' -> '.join(c)}")

# --- GUI components ---

class GraphCanvas(FigureCanvas):
    """ Matplotlib canvas for graph with draggable nodes """
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width,height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111)
        self.G = nx.DiGraph()
        self.pos = {}
        self.node_artist_map = {}
        self.selected_node = None
        self.dragging = False

        self._cid_press = self.mpl_connect('button_press_event', self._on_press)
        self._cid_release = self.mpl_connect('button_release_event', self._on_release)
        self._cid_move = self.mpl_connect('motion_notify_event', self._on_motion)
        self._cid_pick = self.mpl_connect('pick_event', self._on_pick)

    def load_from_fsm(self, fsm: Dict[str, Any]):
        self.G.clear()
        self.node_artist_map = {}
        # nodes
        for s in fsm.get("states", []):
            sid = s.get("id")
            self.G.add_node(sid)
        # edges
        for s in fsm.get("states", []):
            for t in s.get("transitions", []):
                src = s.get("id")
                tgt = t.get("target")
                tid = t.get("id", "")
                if tgt:
                    self.G.add_edge(src, tgt, id=tid, label=tid)
        # compute layout
        if len(self.G.nodes) == 0:
            self.pos = {}
        else:
            self.pos = nx.spring_layout(self.G, seed=2, k=1.2/ (len(self.G.nodes)**0.5))
        self.draw_graph()

    def autolayout(self):
        if len(self.G.nodes) > 0:
            self.pos = nx.spring_layout(self.G, seed=2, k=1.2/ (len(self.G.nodes)**0.5))
            self.draw_graph()

    def draw_graph(self):
        self.ax.clear()
        if len(self.G.nodes) == 0:
            self.draw()
            return
        nx.draw_networkx_edges(self.G, pos=self.pos, ax=self.ax, arrowstyle='->', arrowsize=12)
        # draw nodes as scatter to make them pickable
        xs = [self.pos[n][0] for n in self.G.nodes]
        ys = [self.pos[n][1] for n in self.G.nodes]
        nodesc = self.ax.scatter(xs, ys, s=800, c='lightblue', zorder=3, picker=5)
        # put labels
        for n in self.G.nodes:
            x,y = self.pos[n]
            self.ax.text(x, y, n, horizontalalignment='center', verticalalignment='center', fontsize=9, zorder=4)
        # store mapping of node bounding box region ~ by coordinates
        self.draw()
        # rebuild artist mapping roughly by proximity
        self.node_artist_map = {n: (self.pos[n][0], self.pos[n][1]) for n in self.G.nodes}

    def _closest_node(self, event):
        if event.xdata is None or event.ydata is None:
            return None
        min_d = 9999
        best = None
        for n,p in self.node_artist_map.items():
            dx = p[0] - event.xdata
            dy = p[1] - event.ydata
            d = dx*dx + dy*dy
            if d < min_d:
                min_d = d
                best = n
        # threshold
        if min_d < 0.05:
            return best
        return None

    def _on_press(self, event):
        node = self._closest_node(event)
        if node:
            self.selected_node = node
            self.dragging = True

    def _on_release(self, event):
        self.dragging = False
        self.selected_node = None

    def _on_motion(self, event):
        if not self.dragging or self.selected_node is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.pos[self.selected_node] = (event.xdata, event.ydata)
        self.draw_graph()

    def _on_pick(self, event):
        # pass
        pass

# --- Main Window ---
class FSMConfiguratorMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSM Configurator — Extended")
        self.resize(1200, 800)
        self.current_file: Optional[str] = None
        self.fsm = deepcopy(DEFAULT_EXAMPLE)
        self._build_ui()
        self.refresh_all()

    def _build_ui(self):
        # Toolbar
        toolbar = self.addToolBar("File")
        btn_new = QPushButton("New")
        btn_open = QPushButton("Open")
        btn_save = QPushButton("Save")
        btn_saveas = QPushButton("Save As")
        btn_validate = QPushButton("Run Validation")
        btn_autolayout = QPushButton("Autolayout Graph")
        btn_export_mermaid = QPushButton("Export Mermaid")
        toolbar.addWidget(btn_new); toolbar.addWidget(btn_open); toolbar.addWidget(btn_save); toolbar.addWidget(btn_saveas)
        toolbar.addSeparator()
        toolbar.addWidget(btn_validate); toolbar.addWidget(btn_autolayout); toolbar.addWidget(btn_export_mermaid)
        btn_new.clicked.connect(self.new_file)
        btn_open.clicked.connect(self.open_file)
        btn_save.clicked.connect(self.save_file)
        btn_saveas.clicked.connect(self.save_file_as)
        btn_validate.clicked.connect(self.run_validation)
        btn_autolayout.clicked.connect(self.graph_autolayout)
        btn_export_mermaid.clicked.connect(self.export_mermaid)

        # Main splitter with tabs
        central = QWidget()
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tabs: Metadata, Variables, Sources, Actions, Timers, Global transitions, States, Graph, Validators, Export
        self.tab_meta = QWidget(); self.tab_vars = QWidget(); self.tab_sources = QWidget()
        self.tab_actions = QWidget(); self.tab_timers = QWidget(); self.tab_globals = QWidget()
        self.tab_states = QWidget(); self.tab_graph = QWidget(); self.tab_valid = QWidget(); self.tab_export = QWidget()

        self.tabs.addTab(self.tab_meta, "Metadata")
        self.tabs.addTab(self.tab_vars, "Variables")
        self.tabs.addTab(self.tab_sources, "Sources")
        self.tabs.addTab(self.tab_actions, "Actions")
        self.tabs.addTab(self.tab_timers, "Timers")
        self.tabs.addTab(self.tab_globals, "Global Transitions")
        self.tabs.addTab(self.tab_states, "States")
        self.tabs.addTab(self.tab_graph, "Graph")
        self.tabs.addTab(self.tab_valid, "Validators")
        self.tabs.addTab(self.tab_export, "Export")

        # Metadata tab
        meta_layout = QFormLayout()
        self.tab_meta.setLayout(meta_layout)
        self.input_name = QLineEdit()
        self.input_desc = QLineEdit()
        self.input_initial = QLineEdit()
        self.input_schema = QLineEdit()
        meta_layout.addRow("Name", self.input_name)
        meta_layout.addRow("Description", self.input_desc)
        meta_layout.addRow("Initial State", self.input_initial)
        meta_layout.addRow("Schema Version", self.input_schema)

        # Variables tab
        vlayout = QVBoxLayout()
        self.tab_vars.setLayout(vlayout)
        self.vars_table = QTableWidget(0,5)
        self.vars_table.setHorizontalHeaderLabels(["name","type","initial","persist","description"])
        vlayout.addWidget(self.vars_table)
        v_add = QPushButton("Add Variable")
        v_add.clicked.connect(self.add_variable_row)
        vlayout.addWidget(v_add)

        # Sources tab
        slayout = QVBoxLayout()
        self.tab_sources.setLayout(slayout)
        self.sources_table = QTableWidget(0,4)
        self.sources_table.setHorizontalHeaderLabels(["id","type","connection","description"])
        slayout.addWidget(self.sources_table)
        s_add = QPushButton("Add Source")
        s_add.clicked.connect(self.add_source_row)
        slayout.addWidget(s_add)

        # Actions tab
        alayout = QVBoxLayout()
        self.tab_actions.setLayout(alayout)
        self.actions_table = QTableWidget(0,4)
        self.actions_table.setHorizontalHeaderLabels(["id","kind","type","impl/params"])
        alayout.addWidget(self.actions_table)
        a_add = QPushButton("Add Action")
        a_add.clicked.connect(self.add_action_row)
        alayout.addWidget(a_add)

        # Timers tab
        tlayout = QVBoxLayout()
        self.tab_timers.setLayout(tlayout)
        self.timers_table = QTableWidget(0,6)
        self.timers_table.setHorizontalHeaderLabels(["id","type","interval_ms","event","auto_start","payload"])
        tlayout.addWidget(self.timers_table)
        ta_add = QPushButton("Add Timer")
        ta_add.clicked.connect(self.add_timer_row)
        tlayout.addWidget(ta_add)

        # Global transitions tab
        glayout = QVBoxLayout()
        self.tab_globals.setLayout(glayout)
        self.globals_table = QTableWidget(0,6)
        self.globals_table.setHorizontalHeaderLabels(["id","trigger","guard","target","actions","priority"])
        glayout.addWidget(self.globals_table)
        g_add = QPushButton("Add Global Transition")
        g_add.clicked.connect(self.add_global_row)
        glayout.addWidget(g_add)

        # States tab (left: list, right: card editor)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.tab_states.setLayout(QHBoxLayout())
        self.tab_states.layout().addWidget(split)
        left_widget = QWidget(); left_layout = QVBoxLayout(); left_widget.setLayout(left_layout)
        right_widget = QWidget(); right_layout = QVBoxLayout(); right_widget.setLayout(right_layout)
        split.addWidget(left_widget); split.addWidget(right_widget)
        self.state_list = QListWidget()
        left_layout.addWidget(self.state_list)
        sbtn_layout = QHBoxLayout()
        self.btn_add_state = QPushButton("Add State")
        self.btn_remove_state = QPushButton("Remove State")
        sbtn_layout.addWidget(self.btn_add_state); sbtn_layout.addWidget(self.btn_remove_state)
        left_layout.addLayout(sbtn_layout)
        self.btn_add_state.clicked.connect(self.ui_add_state)
        self.btn_remove_state.clicked.connect(self.ui_remove_state)
        self.state_list.currentItemChanged.connect(self.load_state_card)

        # Right: card editor
        self.card_group = QGroupBox("State Editor")
        right_layout.addWidget(self.card_group)
        self.card_layout = QFormLayout()
        self.card_group.setLayout(self.card_layout)
        self.input_state_id = QLineEdit()
        self.combo_state_type = QComboBox()
        self.combo_state_type.addItems(["atomic","compound","parallel","final","choice"])
        self.input_state_title = QLineEdit()
        self.input_state_desc = QLineEdit()
        self.input_on_enter = QLineEdit()
        self.input_on_exit = QLineEdit()
        self.card_layout.addRow("id", self.input_state_id)
        self.card_layout.addRow("type", self.combo_state_type)
        self.card_layout.addRow("title", self.input_state_title)
        self.card_layout.addRow("description", self.input_state_desc)
        self.card_layout.addRow("on_enter (csv)", self.input_on_enter)
        self.card_layout.addRow("on_exit (csv)", self.input_on_exit)

        # transitions table for state
        self.trans_table = QTableWidget(0,6)
        self.trans_table.setHorizontalHeaderLabels(["id","trigger","guard","target","actions","priority"])
        right_layout.addWidget(QLabel("Transitions:"))
        right_layout.addWidget(self.trans_table)
        tbtns = QHBoxLayout()
        self.t_add = QPushButton("Add Transition"); self.t_remove = QPushButton("Remove Transition")
        tbtns.addWidget(self.t_add); tbtns.addWidget(self.t_remove)
        right_layout.addLayout(tbtns)
        self.t_add.clicked.connect(self.ui_add_transition)
        self.t_remove.clicked.connect(self.ui_remove_transition)

        # timeouts table
        right_layout.addWidget(QLabel("Timeouts:"))
        self.timeouts_table = QTableWidget(0,5)
        self.timeouts_table.setHorizontalHeaderLabels(["id","duration_ms","repeat","cancel_on_exit","target"])
        right_layout.addWidget(self.timeouts_table)
        tobtn = QHBoxLayout()
        self.to_add = QPushButton("Add Timeout"); self.to_remove = QPushButton("Remove Timeout")
        tobtn.addWidget(self.to_add); tobtn.addWidget(self.to_remove)
        right_layout.addLayout(tobtn)
        self.to_add.clicked.connect(self.ui_add_timeout)
        self.to_remove.clicked.connect(self.ui_remove_timeout)

        # Graph tab
        vgr = QVBoxLayout(); self.tab_graph.setLayout(vgr)
        self.graph_canvas = GraphCanvas(self.tab_graph, width=8, height=6)
        vgr.addWidget(self.graph_canvas)
        gbtns = QHBoxLayout()
        self.btn_relayout = QPushButton("Autolayout")
        self.btn_export_graph = QPushButton("Export PNG")
        gbtns.addWidget(self.btn_relayout); gbtns.addWidget(self.btn_export_graph)
        vgr.addLayout(gbtns)
        self.btn_relayout.clicked.connect(self.graph_autolayout)
        self.btn_export_graph.clicked.connect(self.export_graph_png)

        # Validators tab
        vbox = QVBoxLayout(); self.tab_valid.setLayout(vbox)
        self.valid_text = QTextEdit(); self.valid_text.setReadOnly(True)
        vbox.addWidget(self.valid_text)
        self.btn_run_valid = QPushButton("Run Validation")
        vbox.addWidget(self.btn_run_valid)
        self.btn_run_valid.clicked.connect(self.run_validation)

        # Export tab
        ex_layout = QVBoxLayout(); self.tab_export.setLayout(ex_layout)
        self.export_text = QTextEdit()
        ex_layout.addWidget(QLabel("Mermaid / PlantUML preview"))
        ex_layout.addWidget(self.export_text)
        ex_buttons = QHBoxLayout()
        self.btn_export_mermaid2 = QPushButton("Generate Mermaid")
        self.btn_export_puml = QPushButton("Generate PlantUML")
        ex_buttons.addWidget(self.btn_export_mermaid2); ex_buttons.addWidget(self.btn_export_puml)
        ex_layout.addLayout(ex_buttons)
        self.btn_export_mermaid2.clicked.connect(self.generate_mermaid)
        self.btn_export_puml.clicked.connect(self.generate_plantuml)

    # --- UI helpers for tables ---
    def _table_push_rows(self, table: QTableWidget, rows: List[List[Any]]):
        table.setRowCount(0)
        for r in rows:
            row = table.rowCount()
            table.insertRow(row)
            for c, val in enumerate(r):
                item = QTableWidgetItem(str(val) if val is not None else "")
                table.setItem(row, c, item)

    def add_variable_row(self):
        r = self.vars_table.rowCount()
        self.vars_table.insertRow(r)
        for c in range(self.vars_table.columnCount()):
            self.vars_table.setItem(r, c, QTableWidgetItem(""))

    def add_source_row(self):
        r = self.sources_table.rowCount()
        self.sources_table.insertRow(r)
        for c in range(self.sources_table.columnCount()):
            self.sources_table.setItem(r, c, QTableWidgetItem(""))

    def add_action_row(self):
        r = self.actions_table.rowCount()
        self.actions_table.insertRow(r)
        for c in range(self.actions_table.columnCount()):
            self.actions_table.setItem(r, c, QTableWidgetItem(""))

    def add_timer_row(self):
        r = self.timers_table.rowCount()
        self.timers_table.insertRow(r)
        for c in range(self.timers_table.columnCount()):
            self.timers_table.setItem(r, c, QTableWidgetItem(""))

    def add_global_row(self):
        r = self.globals_table.rowCount()
        self.globals_table.insertRow(r)
        for c in range(self.globals_table.columnCount()):
            self.globals_table.setItem(r, c, QTableWidgetItem(""))

    # --- State list and card management ---
    def refresh_all(self):
        self._load_meta_to_ui()
        self._load_vars_to_ui()
        self._load_sources_to_ui()
        self._load_actions_to_ui()
        self._load_timers_to_ui()
        self._load_globals_to_ui()
        self._load_states_to_ui()
        self.graph_canvas.load_from_fsm(self.fsm)

    def _load_meta_to_ui(self):
        self.input_name.setText(self.fsm.get("name",""))
        self.input_desc.setText(self.fsm.get("description",""))
        self.input_initial.setText(self.fsm.get("initial_state",""))
        self.input_schema.setText(self.fsm.get("schema_version", ""))

    def _load_vars_to_ui(self):
        rows = []
        for v in self.fsm.get("variables", []):
            rows.append([v.get("name",""), v.get("type",""), v.get("initial",""), v.get("persist",""), v.get("description","")])
        self._table_push_rows(self.vars_table, rows)

    def _load_sources_to_ui(self):
        rows = []
        for s in self.fsm.get("sources", []):
            rows.append([s.get("id",""), s.get("type",""), s.get("connection",""), s.get("description","")])
        self._table_push_rows(self.sources_table, rows)

    def _load_actions_to_ui(self):
        rows = []
        for a in self.fsm.get("actions", []):
            rows.append([a.get("id",""), a.get("kind", a.get("type","")), a.get("type",""), a.get("impl","")])
        self._table_push_rows(self.actions_table, rows)

    def _load_timers_to_ui(self):
        rows = []
        for t in self.fsm.get("timers", []):
            rows.append([t.get("id",""), t.get("type",""), t.get("interval_ms",""), t.get("event",""), t.get("auto_start",False), t.get("payload","")])
        self._table_push_rows(self.timers_table, rows)

    def _load_globals_to_ui(self):
        rows = []
        for g in self.fsm.get("global.transitions", []) if isinstance(self.fsm.get("global.transitions", []), list) else []:
            rows.append([g.get("id",""), g.get("trigger",""), g.get("guard",""), g.get("target",""), ",".join(g.get("actions",[])), g.get("priority","")])
        self._table_push_rows(self.globals_table, rows)

    def _load_states_to_ui(self):
        self.state_list.clear()
        for s in self.fsm.get("states", []):
            item = QListWidgetItem(s.get("id",""))
            self.state_list.addItem(item)

    def ui_add_state(self):
        text, ok = ("NewState", True)
        if ok:
            # ensure unique id
            base = "NewState"
            i = 1
            ids = [s.get("id") for s in self.fsm.get("states", [])]
            candidate = base
            while candidate in ids:
                i+=1; candidate = f"{base}{i}"
            self.fsm.setdefault("states", []).append({"id": candidate, "type": "atomic"})
            self.refresh_all()

    def ui_remove_state(self):
        item = self.state_list.currentItem()
        if not item: return
        sid = item.text()
        # remove
        self.fsm["states"] = [s for s in self.fsm.get("states", []) if s.get("id") != sid]
        # also remove transitions that point to it (warn)
        for s in self.fsm.get("states", []):
            s["transitions"] = [t for t in s.get("transitions", []) if t.get("target") != sid]
        self.refresh_all()

    def load_state_card(self, current: QListWidgetItem, prev: QListWidgetItem):
        if not current: return
        sid = current.text()
        sdef = None
        for s in self.fsm.get("states", []):
            if s.get("id") == sid:
                sdef = s; break
        if not sdef:
            return
        # populate card
        self.input_state_id.setText(sdef.get("id",""))
        self.combo_state_type.setCurrentText(sdef.get("type","atomic"))
        self.input_state_title.setText(sdef.get("title",""))
        self.input_state_desc.setText(sdef.get("description",""))
        self.input_on_enter.setText(",".join(sdef.get("on_enter", [])))
        self.input_on_exit.setText(",".join(sdef.get("on_exit", [])))
        # transitions
        self._table_push_rows(self.trans_table, [[t.get("id",""), t.get("trigger",""), t.get("guard",""), t.get("target",""), ",".join(t.get("actions",[])), t.get("priority","")] for t in sdef.get("transitions", [])])
        # timeouts
        self._table_push_rows(self.timeouts_table, [[to.get("id",""), to.get("duration_ms",""), to.get("repeat",False), to.get("cancel_on_exit", True), to.get("target","")] for to in sdef.get("timeouts", [])])

    def ui_add_transition(self):
        cur = self.state_list.currentItem()
        if not cur:
            QMessageBox.warning(self, "No state", "Choose a state to add transition")
            return
        sid = cur.text()
        sdef = next((s for s in self.fsm.get("states", []) if s.get("id")==sid), None)
        if sdef is None: return
        trow = {"id": f"t_{sid}_new_{len(sdef.get('transitions',[]))+1}", "trigger": "", "guard": "", "target": "", "actions": [], "priority": 10}
        sdef.setdefault("transitions", []).append(trow)
        self.load_state_card(cur, None)

    def ui_remove_transition(self):
        cur = self.state_list.currentItem()
        if not cur: return
        sid = cur.text()
        sdef = next((s for s in self.fsm.get("states", []) if s.get("id")==sid), None)
        if sdef is None: return
        row = self.trans_table.currentRow()
        if row < 0: return
        sdef["transitions"].pop(row)
        self.load_state_card(cur, None)

    def ui_add_timeout(self):
        cur = self.state_list.currentItem()
        if not cur: return
        sid = cur.text()
        sdef = next((s for s in self.fsm.get("states", []) if s.get("id")==sid), None)
        if sdef is None: return
        to = {"id": f"to_{sid}_{len(sdef.get('timeouts',[]))+1}", "duration_ms": 1000, "repeat": False, "cancel_on_exit": True, "target": sid}
        sdef.setdefault("timeouts", []).append(to)
        self.load_state_card(cur, None)

    def ui_remove_timeout(self):
        cur = self.state_list.currentItem()
        if not cur: return
        sid = cur.text()
        sdef = next((s for s in self.fsm.get("states", []) if s.get("id")==sid), None)
        if sdef is None: return
        row = self.timeouts_table.currentRow()
        if row < 0: return
        sdef["timeouts"].pop(row)
        self.load_state_card(cur, None)

    # --- File actions ---
    def new_file(self):
        self.fsm = deepcopy(DEFAULT_EXAMPLE)
        self.current_file = None
        self.refresh_all()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open FSM TOML", "", "TOML files (*.toml)")
        if not path: return
        try:
            self.fsm = load_toml_file(path)
            self.current_file = path
            self.refresh_all()
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load TOML: {e}")

    def save_file(self):
        if not self.current_file:
            return self.save_file_as()
        self._sync_ui_to_model()
        try:
            save_toml_file(self.current_file, self.fsm)
            QMessageBox.information(self, "Saved", f"Saved to {self.current_file}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save TOML: {e}")

    def save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save FSM TOML", "", "TOML files (*.toml)")
        if not path: return
        self.current_file = path
        self.save_file()

    def _sync_ui_to_model(self):
        # Meta
        self.fsm["name"] = self.input_name.text().strip()
        self.fsm["description"] = self.input_desc.text().strip()
        self.fsm["initial_state"] = self.input_initial.text().strip()
        self.fsm["schema_version"] = self.input_schema.text().strip() or DEFAULT_SCHEMA_VERSION
        # variables
        vars_out = []
        for r in range(self.vars_table.rowCount()):
            name = self.vars_table.item(r,0).text() if self.vars_table.item(r,0) else ""
            if not name: continue
            vars_out.append({"name": name, "type": self.vars_table.item(r,1).text() if self.vars_table.item(r,1) else "string",
                             "initial": self.vars_table.item(r,2).text() if self.vars_table.item(r,2) else None,
                             "persist": self.vars_table.item(r,3).text()=="True", "description": self.vars_table.item(r,4).text() if self.vars_table.item(r,4) else ""})
        self.fsm["variables"] = vars_out
        # sources
        srcs = []
        for r in range(self.sources_table.rowCount()):
            idc = self.sources_table.item(r,0).text() if self.sources_table.item(r,0) else ""
            if not idc: continue
            srcs.append({"id": idc, "type": self.sources_table.item(r,1).text() if self.sources_table.item(r,1) else "",
                         "connection": self.sources_table.item(r,2).text() if self.sources_table.item(r,2) else "",
                         "description": self.sources_table.item(r,3).text() if self.sources_table.item(r,3) else ""})
        self.fsm["sources"] = srcs
        # actions
        acts = []
        for r in range(self.actions_table.rowCount()):
            aid = self.actions_table.item(r,0).text() if self.actions_table.item(r,0) else ""
            if not aid: continue
            acts.append({"id": aid, "kind": self.actions_table.item(r,1).text() if self.actions_table.item(r,1) else "",
                         "type": self.actions_table.item(r,2).text() if self.actions_table.item(r,2) else "",
                         "impl": self.actions_table.item(r,3).text() if self.actions_table.item(r,3) else ""})
        self.fsm["actions"] = acts
        # timers
        timers = []
        for r in range(self.timers_table.rowCount()):
            tid = self.timers_table.item(r,0).text() if self.timers_table.item(r,0) else ""
            if not tid: continue
            timers.append({"id": tid, "type": self.timers_table.item(r,1).text() if self.timers_table.item(r,1) else "",
                           "interval_ms": int(self.timers_table.item(r,2).text()) if self.timers_table.item(r,2) and self.timers_table.item(r,2).text().isdigit() else 0,
                           "event": self.timers_table.item(r,3).text() if self.timers_table.item(r,3) else "",
                           "auto_start": self.timers_table.item(r,4).text()=="True", "payload": self.timers_table.item(r,5).text() if self.timers_table.item(r,5) else ""})
        self.fsm["timers"] = timers
        # globals
        gls = []
        for r in range(self.globals_table.rowCount()):
            gid = self.globals_table.item(r,0).text() if self.globals_table.item(r,0) else ""
            if not gid: continue
            gls.append({"id": gid, "trigger": self.globals_table.item(r,1).text() if self.globals_table.item(r,1) else "",
                        "guard": self.globals_table.item(r,2).text() if self.globals_table.item(r,2) else "",
                        "target": self.globals_table.item(r,3).text() if self.globals_table.item(r,3) else "",
                        "actions": [x.strip() for x in (self.globals_table.item(r,4).text() if self.globals_table.item(r,4) else "").split(",") if x.strip()],
                        "priority": int(self.globals_table.item(r,5).text()) if self.globals_table.item(r,5) and self.globals_table.item(r,5).text().isdigit() else 0})
        self.fsm["global.transitions"] = gls
        # states: need to sync edited state
        # sync card edits back to currently selected state
        cur = self.state_list.currentItem()
        if cur:
            sid = cur.text()
            sdef = next((s for s in self.fsm.get("states", []) if s.get("id")==sid), None)
            if sdef:
                sdef["id"] = self.input_state_id.text().strip() or sdef["id"]
                sdef["type"] = self.combo_state_type.currentText()
                sdef["title"] = self.input_state_title.text().strip()
                sdef["description"] = self.input_state_desc.text().strip()
                sdef["on_enter"] = [x.strip() for x in self.input_on_enter.text().split(",") if x.strip()]
                sdef["on_exit"] = [x.strip() for x in self.input_on_exit.text().split(",") if x.strip()]
                # transitions
                trans = []
                for r in range(self.trans_table.rowCount()):
                    tid = self.trans_table.item(r,0).text() if self.trans_table.item(r,0) else f"t_{sdef['id']}_{r}"
                    trig = self.trans_table.item(r,1).text() if self.trans_table.item(r,1) else ""
                    guard = self.trans_table.item(r,2).text() if self.trans_table.item(r,2) else ""
                    target = self.trans_table.item(r,3).text() if self.trans_table.item(r,3) else ""
                    acts = [x.strip() for x in (self.trans_table.item(r,4).text() if self.trans_table.item(r,4) else "").split(",") if x.strip()]
                    pr = int(self.trans_table.item(r,5).text()) if self.trans_table.item(r,5) and self.trans_table.item(r,5).text().isdigit() else 0
                    trans.append({"id": tid, "trigger": trig, "guard": guard, "target": target, "actions": acts, "priority": pr})
                    # auto-create target state if missing
                    if target and target not in [s.get("id") for s in self.fsm.get("states", [])]:
                        self.fsm.setdefault("states", []).append({"id": target, "type": "atomic"})
                sdef["transitions"] = trans
                # timeouts
                tos = []
                for r in range(self.timeouts_table.rowCount()):
                    tid = self.timeouts_table.item(r,0).text() if self.timeouts_table.item(r,0) else f"to_{sdef['id']}_{r}"
                    dur = int(self.timeouts_table.item(r,1).text()) if self.timeouts_table.item(r,1) and self.timeouts_table.item(r,1).text().isdigit() else 0
                    rep = (self.timeouts_table.item(r,2).text().lower() == "true") if self.timeouts_table.item(r,2) else False
                    coe = (self.timeouts_table.item(r,3).text().lower()=="true") if self.timeouts_table.item(r,3) else True
                    tgt = self.timeouts_table.item(r,4).text() if self.timeouts_table.item(r,4) else sdef.get("id")
                    tos.append({"id": tid, "duration_ms": dur, "repeat": rep, "cancel_on_exit": coe, "target": tgt})
                sdef["timeouts"] = tos
        # Refresh list (in case ids changed)
        self._load_states_to_ui()
        self.graph_canvas.load_from_fsm(self.fsm)

    # --- Validation / Graph / Exports ---
    def run_validation(self):
        self._sync_ui_to_model()
        validator = FSMValidator(deepcopy(self.fsm))
        result = validator.run_all()
        self.valid_text.clear()
        if result["errors"]:
            self.valid_text.append("=== ERRORS ===")
            for e in result["errors"]:
                self.valid_text.append("- " + e)
        if result["warnings"]:
            self.valid_text.append("\n=== WARNINGS ===")
            for w in result["warnings"]:
                self.valid_text.append("- " + w)
        if not result["errors"] and not result["warnings"]:
            self.valid_text.append("No issues found. Validation OK.")
        # Refresh graph
        self.graph_canvas.load_from_fsm(self.fsm)

    def graph_autolayout(self):
        self.graph_canvas.autolayout()

    def export_graph_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Graph PNG", "", "PNG Files (*.png)")
        if not path: return
        self.graph_canvas.figure.savefig(path)
        QMessageBox.information(self, "Export", f"Saved graph to {path}")

    def export_mermaid(self):
        self._sync_ui_to_model()
        text = self._generate_mermaid_text()
        path, _ = QFileDialog.getSaveFileName(self, "Export Mermaid", "", "Markdown/TXT (*.md *.txt)")
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        QMessageBox.information(self, "Export", f"Mermaid exported to {path}")

    def generate_mermaid(self):
        self._sync_ui_to_model()
        self.export_text.setPlainText(self._generate_mermaid_text())

    def generate_plantuml(self):
        self._sync_ui_to_model()
        self.export_text.setPlainText(self._generate_plantuml_text())

    def _generate_mermaid_text(self) -> str:
        lines = ["stateDiagram-v2"]
        for s in self.fsm.get("states", []):
            sid = s.get("id")
            # node line optional
            lines.append(f'    {sid}')
        for s in self.fsm.get("states", []):
            for t in s.get("transitions", []):
                src = s.get("id")
                tgt = t.get("target")
                label = t.get("id","")
                if tgt:
                    lines.append(f'    {src} --> {tgt}: {label}')
        return "\n".join(lines)

    def _generate_plantuml_text(self) -> str:
        lines = ["@startuml", "hide empty description", "skinparam state {\n  BackgroundColor LightYellow\n}"]
        for s in self.fsm.get("states", []):
            lines.append(f'state {s.get("id")}')
        for s in self.fsm.get("states", []):
            for t in s.get("transitions", []):
                src = s.get("id"); tgt = t.get("target"); label = t.get("id","")
                if tgt:
                    lines.append(f'{src} --> {tgt} : {label}')
        lines.append("@enduml")
        return "\n".join(lines)

# --- Main run ---
def main():
    app = QApplication(sys.argv)
    win = FSMConfiguratorMain()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
