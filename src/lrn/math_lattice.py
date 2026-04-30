"""
MathLattice - Extends LatticeNN with mathematical operations
Number line as tensegrity, equations as equilibrium states
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, Node, Spring
from typing import Optional, List, Tuple, Dict


class MathLattice(LatticeNN):
    """LatticeNN with math capabilities - number line and equations"""
    
    def __init__(self):
        super().__init__()
        self.number_line_min = -100
        self.number_line_max = 100
        self._math_initialized = False
    
    def initialize_math(self):
        """Initialize number line and operators"""
        if self._math_initialized:
            return
        
        # Create number nodes on the line
        for i in range(self.number_line_min, self.number_line_max + 1):
            self.add_node(f"sensor:count:{i}")
            self.nodes[f"sensor:count:{i}"].x = i * 100
            self.nodes[f"sensor:count:{i}"].tau = 0  # Constitutive
        
        # Add proximity springs between adjacent numbers
        for i in range(self.number_line_min, self.number_line_max):
            for j in range(i + 1, min(i + 11, self.number_line_max + 1)):
                dist = j - i
                k = 100 // dist
                if k >= 1:
                    self.add_or_update_spring(
                        f"sensor:count:{i}", 
                        f"sensor:count:{j}", 
                        stiffness=k, 
                        tau=1
                    )
        
        # Create operator nodes
        for op in ["plus", "minus", "equals", "times", "divide"]:
            self.add_node(f"op:{op}")
            self.nodes[f"op:{op}"].tau = 0
        
        # Link operator words to operator nodes
        operator_mappings = [
            ("plus", "op:plus"), ("+", "op:plus"),
            ("minus", "op:minus"), ("-", "op:minus"),
            ("equals", "op:equals"), ("=", "op:equals"),
            ("times", "op:times"), ("×", "op:times"),
            ("divide", "op:divide"), ("÷", "op:divide"),
        ]
        for word, op_node in operator_mappings:
            self.add_or_update_spring(word, op_node, stiffness=100, tau=0)
        
        # Create zero as anchor
        self.add_node("sensor:count:0")
        self.nodes["sensor:count:0"].activation = 100
        self.nodes["sensor:count:0"].pinned = True
        
        self._math_initialized = True
        print(f"  [Math] Number line initialized: {self.number_line_max - self.number_line_min + 1} nodes")
    
    def extract_value(self, node_id: str) -> Optional[int]:
        """Extract integer value from 'sensor:count:X' format"""
        if not node_id.startswith("sensor:count:"):
            return None
        try:
            return int(node_id.split(":")[-1])
        except ValueError:
            return None
    
    def is_number_node(self, node_id: str) -> bool:
        """Check if node is a number on the line"""
        val = self.extract_value(node_id)
        return val is not None and self.number_line_min <= val <= self.number_line_max


class NumberLine:
    """Traverses the number line - forward/backward counting"""
    
    def __init__(self, math_lattice: MathLattice):
        self.lnn = math_lattice
    
    def step_forward(self, start_val: int, steps: int, verbose: bool = False) -> int:
        """Count forward on number line"""
        current = start_val
        if verbose:
            print(f"  [Math] Counting forward: {start_val} + {steps}")
        
        for _ in range(steps):
            next_val = current + 1
            if next_val > self.lnn.number_line_max:
                break
            
            target = f"sensor:count:{next_val}"
            if target in self.lnn.nodes:
                self.lnn.nodes[target].activation = 80
                if verbose:
                    print(f"    -> {next_val}")
            current = next_val
        
        return current
    
    def step_backward(self, start_val: int, steps: int, verbose: bool = False) -> int:
        """Count backward on number line (crosses into negatives)"""
        current = start_val
        if verbose:
            print(f"  [Math] Counting backward: {start_val} - {steps}")
        
        for _ in range(steps):
            next_val = current - 1
            if next_val < self.lnn.number_line_min:
                break
            
            target = f"sensor:count:{next_val}"
            if target in self.lnn.nodes:
                self.lnn.nodes[target].activation = 80
                if verbose:
                    print(f"    -> {next_val}")
            current = next_val
        
        return current
    
    def multiply(self, groups: int, size: int, verbose: bool = False) -> int:
        """Multiplication as repeated forward counting"""
        if verbose:
            print(f"  [Math] Multiplying: {groups} groups of {size}")
        
        current = 0
        for i in range(groups):
            if verbose:
                print(f"    Group {i + 1}:")
            current = self.step_forward(current, size, verbose=verbose)
        
        return current
    
    def divide(self, dividend: int, divisor: int, verbose: bool = False) -> Tuple[int, int]:
        """Division as repeated backward counting - returns (quotient, remainder)"""
        if verbose:
            print(f"  [Math] Dividing: {dividend} into groups of {divisor}")
        
        quotient = 0
        current = dividend
        
        while current >= divisor:
            if verbose:
                print(f"    Shard {quotient + 1}:")
            current = self.step_backward(current, divisor, verbose=verbose)
            quotient += 1
        
        remainder = current
        
        if verbose:
            print(f"  [Math] Result: {quotient} remainder {remainder}")
        
        return quotient, remainder


class EquationSolver:
    """Solves equations via tensegrity - same as song completion"""
    
    def __init__(self, math_lattice: MathLattice, number_line: NumberLine):
        self.lnn = math_lattice
        self.nl = number_line
        self.variables: Dict[str, str] = {}
    
    def install_addition_facts(self, max_val: int = 20):
        """Install balance nodes for all addition facts up to max_val"""
        K_BAL_POS = 50
        K_BAL_NEG = -50
        
        for a in range(0, max_val + 1):
            for b in range(0, max_val - a + 1):
                c = a + b
                bal = f"balance:{a}:+:{b}"
                self.lnn.get_or_create(bal)
                
                # Operand springs (push positive)
                self.lnn.add_or_update_spring(
                    f"sensor:count:{a}", bal, stiffness=K_BAL_POS, tau=0
                )
                self.lnn.add_or_update_spring(
                    f"sensor:count:{b}", bal, stiffness=K_BAL_POS, tau=0
                )
                self.lnn.add_or_update_spring(
                    "op:plus", bal, stiffness=K_BAL_POS // 2, tau=0
                )
                
                # Correct answer (negative - cancellation)
                self.lnn.add_or_update_spring(
                    f"sensor:count:{c}", bal, stiffness=K_BAL_NEG, tau=0, mode="neg_override"
                )
                
                # Wrong answers (positive - adds tension)
                for wrong in range(max_val + 1):
                    if wrong != c:
                        self.lnn.add_or_update_spring(
                            f"sensor:count:{wrong}", bal, stiffness=K_BAL_POS, tau=0
                        )
        
        balance_count = len([n for n in self.lnn.nodes if n.startswith("balance:")])
        print(f"  [Math] Installed {balance_count} balance nodes")
    
    def install_multiplication_facts(self, max_val: int = 12):
        """Install balance nodes for multiplication"""
        K_BAL_POS = 50
        K_BAL_NEG = -50
        
        for a in range(0, max_val + 1):
            for b in range(0, max_val + 1):
                c = a * b
                if c > 100:  # Stay within number line range
                    continue
                bal = f"balance:{a}:x:{b}"
                self.lnn.get_or_create(bal)
                
                # Operand springs (push positive)
                self.lnn.add_or_update_spring(
                    f"sensor:count:{a}", bal, stiffness=K_BAL_POS, tau=0
                )
                self.lnn.add_or_update_spring(
                    f"sensor:count:{b}", bal, stiffness=K_BAL_POS, tau=0
                )
                self.lnn.add_or_update_spring(
                    "op:times", bal, stiffness=K_BAL_POS // 2, tau=0
                )
                
                # Correct answer (negative - cancellation)
                self.lnn.add_or_update_spring(
                    f"sensor:count:{c}", bal, stiffness=K_BAL_NEG, tau=0, mode="neg_override"
                )
                
                # Wrong answers (positive - adds tension)
                for wrong in range(0, 101):
                    if wrong != c:
                        self.lnn.add_or_update_spring(
                            f"sensor:count:{wrong}", bal, stiffness=K_BAL_POS, tau=0
                        )
        
        print(f"  [Math] Multiplication facts installed")
    
    def install_division_facts(self, max_val: int = 12):
        """Install balance nodes for division"""
        K_BAL_POS = 50
        K_BAL_NEG = -50
        
        for dividend in range(1, max_val + 1):
            for divisor in range(1, max_val + 1):
                quotient = dividend // divisor
                remainder = dividend % divisor
                bal = f"balance:{dividend}:div:{divisor}"
                self.lnn.get_or_create(bal)
                
                # Dividend and divisor push positive
                self.lnn.add_or_update_spring(
                    f"sensor:count:{dividend}", bal, stiffness=K_BAL_POS, tau=0
                )
                self.lnn.add_or_update_spring(
                    f"sensor:count:{divisor}", bal, stiffness=K_BAL_POS, tau=0
                )
                self.lnn.add_or_update_spring(
                    "op:divide", bal, stiffness=K_BAL_POS // 2, tau=0
                )
                
                # Correct quotient (negative - cancellation)
                self.lnn.add_or_update_spring(
                    f"sensor:count:{quotient}", bal, stiffness=K_BAL_NEG, tau=0, mode="neg_override"
                )
                
                # Wrong answers
                for wrong in range(0, 101):
                    if wrong != quotient:
                        self.lnn.add_or_update_spring(
                            f"sensor:count:{wrong}", bal, stiffness=K_BAL_POS, tau=0
                        )
        
        print(f"  [Math] Division facts installed")
    
    def solve_algebraic_equation(self, eq_type: str, a: int, b: int, result: int, verbose: bool = False) -> int:
        """
        Solve algebraic equations via balance node tension minimization.
        Returns the unknown value.
        
        Types:
        - "add": a + x = result -> x = result - a
        - "sub": x - a = result -> x = result + a  
        - "mul": a * x = result -> x = result / a
        - "div": x / a = result -> x = result * a
        """
        from lrn import propagate
        
        best_answer = 0
        min_tension = float('inf')
        
        # Test each possible answer in range
        for candidate in range(-100, 101):
            self.lnn.reset()
            
            # Pin known values
            if eq_type in ["add", "sub", "mul", "div"]:
                if a is not None:
                    self.lnn.add_node(f"sensor:count:{a}")
                    self.lnn.nodes[f"sensor:count:{a}"].activation = 100
                    self.lnn.nodes[f"sensor:count:{a}"].pinned = True
                
                if b is not None:
                    self.lnn.add_node(f"sensor:count:{b}")
                    self.lnn.nodes[f"sensor:count:{b}"].activation = 100
                    self.lnn.nodes[f"sensor:count:{b}"].pinned = True
            
            # Pin result
            self.lnn.add_node(f"sensor:count:{result}")
            self.lnn.nodes[f"sensor:count:{result}"].activation = 100
            self.lnn.nodes[f"sensor:count:{result}"].pinned = True
            
            # Candidate answer
            self.lnn.add_node(f"sensor:count:{candidate}")
            self.lnn.nodes[f"sensor:count:{candidate}"].activation = 80
            
            # Propagate
            for _ in range(10):
                propagate(self.lnn, n_steps=1)
            
            # Get tension from appropriate balance node
            if eq_type == "add":
                bal = f"balance:{a}:+:{b}"
            elif eq_type == "sub":
                bal = f"balance:{result}:-:b" if a is None else f"balance:{a}:-:b"
            elif eq_type == "mul":
                bal = f"balance:{a}:x:{b}"
            elif eq_type == "div":
                bal = f"balance:{result}:div:{b}"
            else:
                continue
            
            if bal in self.lnn.nodes:
                tension = 0
                for neighbor, sp in self.lnn.get_neighbors(bal):
                    if neighbor in self.lnn.nodes:
                        tension += sp.stiffness * self.lnn.nodes[neighbor].activation
                
                if abs(tension) < abs(min_tension):
                    min_tension = tension
                    best_answer = candidate
        
        if verbose:
            print(f"  [Math] Solved {eq_type}: {a} ? {b} = {result} -> {best_answer}")
        
        return best_answer
    
    def install_subtraction_facts(self, max_val: int = 20):
        """Install balance nodes for subtraction"""
        K_BAL_POS = 50
        K_BAL_NEG = -50
        
        for a in range(0, max_val + 1):
            for b in range(0, a + 1):
                c = a - b
                bal = f"balance:{a}:-:{b}"
                self.lnn.get_or_create(bal)
                
                self.lnn.add_or_update_spring(
                    f"sensor:count:{a}", bal, stiffness=K_BAL_POS, tau=0
                )
                self.lnn.add_or_update_spring(
                    f"sensor:count:{b}", bal, stiffness=K_BAL_POS, tau=0
                )
                self.lnn.add_or_update_spring(
                    "op:minus", bal, stiffness=K_BAL_POS // 2, tau=0
                )
                
                self.lnn.add_or_update_spring(
                    f"sensor:count:{c}", bal, stiffness=K_BAL_NEG, tau=0, mode="neg_override"
                )
                
                for wrong in range(-20, max_val + 1):
                    if wrong != c:
                        self.lnn.add_or_update_spring(
                            f"sensor:count:{wrong}", bal, stiffness=K_BAL_POS, tau=0
                        )
        
        print(f"  [Math] Subtraction facts installed")
    
    def bind_variable(self, var_name: str, sensor_node: str):
        """Bind a variable to a number node with stiff spring"""
        if not var_name.startswith("var:"):
            var_name = f"var:{var_name}"
        
        self.variables[var_name] = sensor_node
        self.lnn.add_spring(var_name, sensor_node, stiffness=100.0)
        print(f"  [Math] Bound {var_name} to {sensor_node}")
    
    def solve_addition(self, a: int, b: int, verbose: bool = True) -> int:
        """Solve 'a + ? = result' - find result via traversal"""
        from lrn import propagate
        
        self.lnn.reset()
        
        # Pin operand
        a_node = f"sensor:count:{a}"
        self.lnn.add_node(a_node)
        self.lnn.nodes[a_node].activation = 100
        self.lnn.nodes[a_node].pinned = True
        
        # Pin operator
        self.lnn.add_node("op:plus")
        self.lnn.nodes["op:plus"].activation = 100
        self.lnn.nodes["op:plus"].pinned = True
        
        # Test each possible answer
        best_answer = a  # Start with wrong
        min_tension = float('inf')
        
        for candidate in range(self.lnn.number_line_min, self.lnn.number_line_max + 1):
            # Reset
            self.lnn.reset()
            self.lnn.nodes[a_node].activation = 100
            self.lnn.nodes[a_node].pinned = True
            self.lnn.nodes["op:plus"].activation = 100
            self.lnn.nodes["op:plus"].pinned = True
            
            # Try candidate
            cand_node = f"sensor:count:{candidate}"
            self.lnn.add_node(cand_node)
            self.lnn.nodes[cand_node].activation = 80
            
            # Propagate
            for _ in range(8):
                propagate(self.lnn, n_steps=1)
            
            # Measure tension on balance node
            bal = f"balance:{a}:+:{b}"
            if bal in self.lnn.nodes:
                bal_node = self.lnn.nodes[bal]
                tension = sum(
                    sp.stiffness * self.lnn.nodes.get(n, Node("")).activation
                    for n, sp in self.lnn.get_neighbors(bal)
                    if n in self.lnn.nodes
                )
                
                if abs(tension) < abs(min_tension):
                    min_tension = tension
                    best_answer = candidate
        
        if verbose:
            print(f"  [Math] Solved: {a} + ? = {a + b}, answer = {best_answer}")
        
        return best_answer
    
    def evaluate_addition(self, a: int, b: int, candidate: int) -> int:
        """Return tension for a + b = candidate"""
        from lrn import propagate
        
        self.lnn.reset()
        
        # Pin all known terms
        for val, name in [(a, "a"), (b, "b"), (candidate, "c")]:
            node_id = f"sensor:count:{val}"
            self.lnn.add_node(node_id)
            self.lnn.nodes[node_id].activation = 100
            self.lnn.nodes[node_id].pinned = True
        
        # Propagate
        for _ in range(8):
            propagate(self.lnn, n_steps=1)
        
        # Check balance node
        bal = f"balance:{a}:+:{b}"
        if bal not in self.lnn.nodes:
            return 999
        
        tension = 0
        for neighbor, sp in self.lnn.get_neighbors(bal):
            if neighbor in self.lnn.nodes:
                tension += sp.stiffness * self.lnn.nodes[neighbor].activation
        
        return tension


def create_math_lattice() -> MathLattice:
    """Factory to create and initialize math lattice"""
    ml = MathLattice()
    ml.initialize_math()
    nl = NumberLine(ml)
    es = EquationSolver(ml, nl)
    es.install_addition_facts(20)
    es.install_subtraction_facts(20)
    return ml