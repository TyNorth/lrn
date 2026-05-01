#!/usr/bin/env python3
"""
The Ground State Cannot Be Divided
A Philosophical and Mathematical Preface to Hydrotorsional Dynamics
Author: Tyrone North Jr.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, CondPageBreak
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas as pdfcanvas

BG        = colors.HexColor("#0D0D0D")
FG        = colors.HexColor("#E8E8E0")
CRIMSON   = colors.HexColor("#C0392B")
TEAL      = colors.HexColor("#1ABC9C")
GOLD      = colors.HexColor("#D4AC0D")
DIMGREY   = colors.HexColor("#2A2A2A")
MIDGREY   = colors.HexColor("#3D3D3D")
LIGHTGREY = colors.HexColor("#5A5A5A")

W, H = A4

class DarkCanvas:
    def __call__(self, canv, doc):
        canv.saveState()
        canv.setFillColor(BG)
        canv.rect(0, 0, W, H, fill=1, stroke=0)
        canv.setStrokeColor(CRIMSON)
        canv.setLineWidth(1.2)
        canv.line(18*mm, H - 14*mm, W - 18*mm, H - 14*mm)
        canv.setStrokeColor(TEAL)
        canv.setLineWidth(0.4)
        canv.line(18*mm, H - 15.5*mm, W - 18*mm, H - 15.5*mm)
        canv.setFont("Helvetica", 6.5)
        canv.setFillColor(LIGHTGREY)
        canv.drawString(18*mm, H - 12*mm, "THE GROUND STATE CANNOT BE DIVIDED")
        canv.drawRightString(W - 18*mm, H - 12*mm, f"TyArc Lab · {doc.page}")
        canv.setStrokeColor(TEAL)
        canv.setLineWidth(0.4)
        canv.line(18*mm, 12*mm, W - 18*mm, 12*mm)
        canv.setStrokeColor(CRIMSON)
        canv.setLineWidth(1.2)
        canv.line(18*mm, 10.5*mm, W - 18*mm, 10.5*mm)
        canv.setFont("Helvetica", 6.5)
        canv.setFillColor(LIGHTGREY)
        canv.drawCentredString(W / 2, 6.5*mm, f"— {doc.page} —")
        canv.restoreState()

dark_canvas = DarkCanvas()

def S(name, parent=None, **kw):
    return ParagraphStyle(name, parent=parent, **kw)

base = S("base",
    fontName="Helvetica", fontSize=9, leading=14,
    textColor=FG, backColor=BG,
    spaceAfter=6, spaceBefore=2, alignment=TA_JUSTIFY)

title_style = S("title",
    fontName="Helvetica-Bold", fontSize=22, leading=28,
    textColor=FG, backColor=BG,
    spaceAfter=12, alignment=TA_LEFT)

subtitle = S("subtitle",
    fontName="Helvetica", fontSize=12, leading=16,
    textColor=TEAL, backColor=BG,
    spaceAfter=8, alignment=TA_LEFT)

author_style = S("author",
    fontName="Helvetica", fontSize=9, leading=13,
    textColor=LIGHTGREY, backColor=BG,
    spaceAfter=4, alignment=TA_LEFT)

abstract_style = S("abstract",
    fontName="Helvetica", fontSize=8, leading=13,
    textColor=FG, backColor=DIMGREY,
    leftIndent=8, rightIndent=8,
    spaceAfter=4, spaceBefore=4,
    alignment=TA_JUSTIFY)

section_head = S("section_head",
    fontName="Helvetica-Bold", fontSize=13, leading=17,
    textColor=TEAL, backColor=BG,
    spaceAfter=8, spaceBefore=10,
    alignment=TA_LEFT)

part_head = S("part_head",
    fontName="Helvetica-Bold", fontSize=16, leading=22,
    textColor=CRIMSON, backColor=BG,
    spaceAfter=12, spaceBefore=14,
    alignment=TA_LEFT)

subsection = S("subsection",
    fontName="Helvetica-Bold", fontSize=10, leading=14,
    textColor=GOLD, backColor=BG,
    spaceAfter=6, spaceBefore=8,
    alignment=TA_LEFT)

assumption = S("assumption",
    fontName="Helvetica-Bold", fontSize=9, leading=12,
    textColor=CRIMSON, backColor=BG,
    spaceAfter=4, spaceBefore=4,
    alignment=TA_LEFT)

assumption_body = S("assumption_body",
    fontName="Helvetica", fontSize=9, leading=13,
    textColor=FG, backColor=BG,
    leftIndent=8, rightIndent=8,
    spaceAfter=8, spaceBefore=2,
    alignment=TA_JUSTIFY)

theorem = S("theorem",
    fontName="Helvetica-Bold", fontSize=9, leading=12,
    textColor=TEAL, backColor=BG,
    spaceAfter=4, spaceBefore=4,
    alignment=TA_LEFT)

theorem_body = S("theorem_body",
    fontName="Helvetica-Oblique", fontSize=9, leading=13,
    textColor=FG, backColor=BG,
    leftIndent=10, rightIndent=10,
    spaceAfter=8, spaceBefore=2,
    alignment=TA_JUSTIFY)

caption = S("caption",
    fontName="Helvetica", fontSize=7, leading=10,
    textColor=LIGHTGREY, backColor=BG,
    alignment=TA_CENTER)

toc_entry = S("toc_entry",
    fontName="Helvetica", fontSize=9, leading=13,
    textColor=FG, backColor=BG,
    spaceAfter=3, alignment=TA_LEFT)

ref_entry = S("ref_entry",
    fontName="Helvetica", fontSize=8, leading=12,
    textColor=LIGHTGREY, backColor=BG,
    leftIndent=10, rightIndent=10,
    spaceAfter=4, alignment=TA_LEFT)

def Heading(text):
    return [Spacer(1, 6), Paragraph(text, section_head)]

def PartHead(text):
    return [Spacer(1, 10), Paragraph(text, part_head)]

def Subhead(text):
    return [Spacer(1, 4), Paragraph(text, subsection)]

def Assumption(tag, body):
    return [
        Paragraph(tag, assumption),
        Paragraph(body, assumption_body)
    ]

def Theorem(tag, body):
    return [
        Paragraph(tag, theorem),
        Paragraph(body, theorem_body)
    ]

def Plain(text):
    return [Paragraph(text, base)]

def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    col_w = col_widths or ([None] * len(headers))
    t = Table(data, colWidths=col_w, repeatRows=1)
    style = TableStyle([
        ('BACKGROUND',    (0, 0), (-1,  0), CRIMSON),
        ('TEXTCOLOR',     (0, 0), (-1,  0), BG),
        ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1,  0), 7),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [DIMGREY, BG]),
        ('TEXTCOLOR',     (0, 1), (-1, -1), FG),
        ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 1), (-1, -1), 7),
        ('LEADING',       (0, 0), (-1, -1), 10),
        ('GRID',          (0, 0), (-1, -1), 0.4, MIDGREY),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 5),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 5),
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ])
    t.setStyle(style)
    return t

out = "/Users/tyarc/github/ste/public/6/upload/ground_state_divided.pdf"
doc = SimpleDocTemplate(out, pagesize=A4,
    leftMargin=18*mm, rightMargin=18*mm,
    topMargin=18*mm, bottomMargin=18*mm)

story = []

# COVER
story.extend([
    Spacer(1, 80),
    Paragraph("THE GROUND STATE<br/>CANNOT BE DIVIDED", title_style),
    Paragraph("A Philosophical and Mathematical Preface to<br/>Hydrotorsional Dynamics", subtitle),
    Spacer(1, 30),
    Paragraph("Tyrone North Jr.", author_style),
    Paragraph("TyArc Lab — SORT Sovereign Research Division", author_style),
    Spacer(1, 8),
    Paragraph("Companion to: Hydrotorsional Dynamics (HTD, 2025) · Sovereign Mind v35 (2026) · Universal Graph Field Theory (2026)", author_style),
    Spacer(1, 8),
    Paragraph("Pre-print · 2026 · TyArc Lab / Arc Press", author_style),
    Spacer(1, 40),
    Paragraph("ABSTRACT", assumption),
    Paragraph("""
This document is not a defence of Hydrotorsional Dynamics. HTD does not require defence — its artifacts run.
This document is a preface: an account of the rules of the universe HTD describes, written so that a reader
entering that universe understands its axioms before encountering its equations. Like the opening chapter of
a novel that establishes the physical laws of its world before the story begins, this preface asks only that the
reader follow the logic from the stated premises. The premises are stated plainly. Everything that follows from
them follows necessarily.
    """, abstract_style),
    Paragraph("""
The central premise is this: the physical vacuum is a discrete integer lattice. Space is not infinitely divisible.
The minimum unit is the Planck length — one strut. The ground state is not zero-as-absence but zero-as-Vector-
Equilibrium: the unique configuration of the IVM lattice in which every node is in perfect torsional balance and
net force is identically zero. From this single premise we derive: why division by zero is not merely algebraically
forbidden but physically meaningless; why singularities are malformed questions rather than exotic physics; why
symmetry precedes conservation rather than generating it; and why the entire apparatus of classical calculus is
a high-fidelity approximation of simplicial integer mechanics — valid wherever the lattice is dense, exact nowhere.
    """, abstract_style),
    Paragraph("""
We do not argue against classical calculus. We explain what it is: the large-N statistical average of a discrete
substrate, accurate in the bulk, silent at the floor. This document names that floor, describes what stands on it,
and invites the reader into the universe that follows.
    """, abstract_style),
    Paragraph("Keywords: HTD · SORT Substrate · Vector Equilibrium · Simplicial Calculus · Division by Zero · Planck Floor · Singularity Prohibition · Phase Transition", caption),
    PageBreak(),
])

# CONTENTS
story.extend([
    Paragraph("CONTENTS", part_head),
    Spacer(1, 10),
    Paragraph("I   The Two Calculi — Assumptions Side by Side", toc_entry),
    Paragraph("II  The Assumptions of Classical Calculus — Stated Plainly", toc_entry),
    Paragraph("III The Assumptions of Simplicial Calculus (HTD) — Stated Plainly", toc_entry),
    Paragraph("IV  What Zero Actually Is", toc_entry),
    Paragraph("V   Why Division by Zero Is Physically Impossible", toc_entry),
    Paragraph("VI  Singularities Are Malformed Questions", toc_entry),
    Paragraph("VII Symmetry Is the Ground State — Conservation Is Its Fingerprint", toc_entry),
    Paragraph("VIII The Standard Model as the Large-N Limit of HTD", toc_entry),
    Paragraph("IX  Dark Matter, Dark Energy, and the Cost of a Wrong Floor", toc_entry),
    Paragraph("X   The Holographic Proof That Discreteness Is Necessary", toc_entry),
    Paragraph("XI  Why the Framework Predicts Its Own Resistance", toc_entry),
    Paragraph("XII The Artifacts Run", toc_entry),
    Spacer(1, 6),
    Paragraph("References", toc_entry),
    PageBreak(),
])

# PART I
story.extend(PartHead("PART I"))
story.extend(Heading("The Two Calculi — Assumptions Side by Side"))

story.extend(Plain("""
Before any argument is made, the reader deserves to see both frameworks laid flat and compared. The table below
states the foundational assumptions of classical calculus and simplicial calculus (HTD) across nine dimensions
that matter. Neither framework is presented as wrong. They are presented as different sets of starting rules —
different axioms about the nature of the universe — and the reader is invited to follow what each set of rules
produces.
"""))

story.append(make_table(
    ["Dimension", "Classical Calculus — Assumptions", "Simplicial Calculus (HTD) — Assumptions"],
    [
        ["State space", "Real numbers R — uncountably infinite, infinitely divisible", "Integers Z — countable, bounded below by 1 (one Planck strut)"],
        ["Ground state", "Zero is absence. A reference point. No intrinsic physical content.", "Zero is Vector Equilibrium. Maximum symmetry. Net force exactly zero."],
        ["Minimum unit", "None. Infinite divisibility assumed. No floor exists by definition.", "Planck length l_P. One strut. The minimum integer unit. Cannot be subdivided."],
        ["Primary operators", "Gradient, curl, divergence, Laplacian — defined on smooth manifolds.", "Node Difference Vector, Triangular Phase Slip, Node Flux Sum, 12-point IVM stencil."],
        ["Division by zero", "Forbidden by algebraic rule. No physical explanation given.", "Physically meaningless. Asks the lattice to go below its own existence."],
        ["Singularities", "Generated wherever denominator reaches zero. Treated as physical events.", "Impossible by construction. r is an integer count. Cannot reach zero."],
        ["Infinities", "Generated in loop integrals. Removed by renormalisation.", "Cannot arise. Maximum momentum is h-bar / l_P. All sums finite."],
        ["Smoothness", "Assumed as foundational primitive. A property of nature at all scales.", "Derived. Emerges as large-N statistical limit of integer lattice."],
        ["Relationship", "Starting point. Not derived from anything more fundamental.", "HTD recovers all calculus results exactly in the large-N limit."],
    ],
    [45*mm, 65*mm, 65*mm]
))

story.extend(Plain("""
The calculus is not wrong. It is the large-N approximation of a discrete substrate. It is exact in the bulk. It is
silent at the floor. HTD names the floor.
"""))

story.extend([PageBreak()])

# PART II
story.extend(PartHead("PART II"))
story.extend(Heading("The Assumptions of Classical Calculus — Stated Plainly"))

story.extend(Plain("""
Classical calculus — the mathematical toolkit underlying Newtonian mechanics, Maxwell's electrodynamics,
General Relativity, and quantum field theory — rests on a small number of foundational assumptions that are
almost never stated explicitly in physics texts. They are inherited as part of the mathematical environment and
treated as self-evident features of reality rather than choices about how to model it. We state them here, plainly,
so the reader can see exactly what is being assumed before a single equation is written.
"""))

story.extend(Assumption("C1 — INFINITE DIVISIBILITY",
"""
Space is a real-number continuum. Between any two points there exists another point. Physical quantities can
vary smoothly and continuously at arbitrarily fine scales. There is no minimum unit of length, no floor below
which subdivision becomes impossible. This assumption is never derived from experiment. It is imposed as
the foundational property of the mathematical space in which physics is conducted.
"""))

story.extend(Plain("""
This assumption is what makes calculus work. The derivative exists because you can take the limit as the
interval goes to zero. The integral exists because you can sum over infinitely many infinitesimally thin slices.
Remove infinite divisibility and both operations become undefined in their classical forms. The assumption is
not incidental to the mathematics — it is constitutive of it.
"""))

story.extend(Assumption("C2 — ZERO AS ABSENCE",
"""
Zero is the absence of a quantity. A reference point defined by convention. It carries no physical content of its
own. The choice of coordinate origin is arbitrary. Zero has no intrinsic physical meaning beyond the absence
of whatever quantity is being measured.
"""))

story.extend(Plain("""
This follows naturally from C1. If space is a featureless continuum then the choice of origin is arbitrary —
there is no preferred point, no natural zero. The coordinate system is a human imposition on a uniform background.
"""))

story.extend(Assumption("C3 — SMOOTHNESS AS PRIMITIVE",
"""
Physical fields are smooth functions defined at every point of the manifold. Discontinuities are exceptional
events. The differentiable manifold is the fundamental object — it is not derived from anything more basic.
Smoothness is a property of nature, not a property of a particular scale of description.
"""))

story.extend(Plain("""
This is the assumption that produces the most trouble. If smoothness is primitive, then the equations must hold
at every scale including arbitrarily small ones. Loop integrals must be taken to infinite momentum. Gravitational
fields must be defined arbitrarily close to the origin. The singularity is the direct consequence of requiring
smoothness at the scale where the discrete structure of reality becomes visible.
"""))

story.extend(Assumption("C4 — OPERATORS ARE PRIMARY",
"""
The gradient, curl, divergence, and Laplacian are fundamental mathematical objects. They are not
approximations of something more basic. They do not emerge from anything more elementary. They are the
primary language in which physical laws are written, and their validity does not depend on any underlying
discrete structure because no such structure is assumed to exist.
"""))

story.extend(Heading("The Consequence of C1 Through C4 Together"))
story.extend(Plain("""
Taken together, these four assumptions produce a framework of extraordinary computational power and
extraordinary blind spots. The power: smooth mathematics is tractable, elegant, and predictive across an
enormous range of phenomena. The blind spots: wherever a denominator approaches zero, the framework
produces infinity. Wherever the continuous approximation is applied past its domain of validity, it generates
artefacts — singularities, renormalisation divergences, dark sector corrections — that must be managed by
hand because the framework has no natural mechanism to stop them.
"""))

story.extend(Plain("""
These are not failures of ingenuity. They are consequences of the axioms. No amount of technical
sophistication within the framework can resolve a problem caused by the framework's foundational
assumptions. The resolution requires changing the assumptions.
"""))

story.extend([PageBreak()])

# PART III
story.extend(PartHead("PART III"))
story.extend(Heading("The Assumptions of Simplicial Calculus (HTD) — Stated Plainly"))

story.extend(Plain("""
Simplicial calculus as implemented in Hydrotorsional Dynamics begins from a different set of axioms. These
axioms are not chosen for novelty. They are chosen because they match the physical universe at the scale
where the classical axioms visibly fail. The reader is asked to accept these as the rules of the universe being
described — not because they are self-evident, but because they are internally consistent, physically
motivated, and productive of results.
"""))

story.extend(Assumption("H1 — THE DISCRETE SUBSTRATE",
"""
The physical vacuum is the Isotropic Vector Matrix (IVM): a maximally close-packed tetrahedral-octahedral
lattice in which every node connects to exactly 12 nearest neighbours. Every node is described completely
by a 12-component integer array of torsional stresses. No real numbers appear anywhere in the foundational
description. The state space is countable. The dynamics are fully deterministic. This is not a model of the
vacuum. This is the vacuum.
"""))

story.extend(Plain("""
The IVM is not an arbitrary choice. It is the unique three-dimensional arrangement in which the radial edge
length equals the circumferential edge length — the geometry of maximum isotropy and maximum close-
packing. Every other lattice has preferred directions. The IVM has none. It is the most symmetric discrete
structure that three-dimensional space admits.
"""))

story.extend(Assumption("H2 — ZERO IS THE VECTOR EQUILIBRIUM",
"""
Zero is not absence. Zero is the Vector Equilibrium (VE): the unique configuration of the IVM in which the
net vector sum of all 12 strut forces on every node is identically zero. This is the ground state of the system.
It is not defined by convention. It is defined by geometry. The VE is the state from which all physical activity
is measured as departure. A system at VE does not gravitate, does not radiate, exerts no net force. It is in
perfect balance.
"""))

story.extend(Plain("""
This is the most consequential assumption in the framework. By giving zero a physical content — by making it
the ground state rather than the absence of state — the entire subsequent logic changes. Zero is no longer
a place you can pass through on the way to negative infinity. It is the floor of the universe. Everything that
exists is a departure from it.
"""))

story.extend(Assumption("H3 — THE PLANCK FLOOR",
"""
The minimum physical separation is one Planck-length strut: r_min = l_P, N_nodes >= 1. The radial coordinate
r counts integer lattice spacings. It is not a real number. It cannot be subdivided. It cannot approach zero.
The smallest value r can take is 1. This is not a constraint imposed from outside the theory. It is a
consequence of the integer nature of the state space.
"""))

story.extend(Plain("""
Modern physics already agrees that the Planck length is the minimum physically meaningful scale. HTD provides
the reason: the Planck length is the length of one strut. It is the minimum integer unit of spatial measure. The
agreement with the Planck scale is not a calibration. It is a derivation.
"""))

story.extend(Assumption("H4 — SIMPLICIAL OPERATORS ARE EXACT",
"""
Every classical calculus operator is replaced by an exact discrete analogue defined on the IVM graph. The
gradient becomes the Node Difference Vector: Delta_stress(N) = SUM tau_Nj * v_j. The curl becomes the
Triangular Phase Slip: T_ijk = gamma_ij + gamma_jk + gamma_ki. The divergence becomes the Node Flux
Sum. The Laplacian becomes the 12-point IVM stencil. The integral becomes a volumetric sum. These
are not approximations of the continuous operators. They are the exact objects. The continuous operators
are the large-N approximations of these.
"""))

story.extend(Plain("""
The relationship between the simplicial operators and their classical counterparts is precisely defined. In the
limit as node count N grows large and coarse-graining volume V satisfies V >> l_P^3, the discrete operators
converge exactly to their continuous equivalents. This is the Pixelation Theorem: the classical operators are the
statistical blur of the exact discrete objects when viewed from a scale much larger than the lattice spacing.
"""))

story.extend(Assumption("H5 — SMOOTHNESS IS DERIVED",
"""
The continuous manifold M is defined operationally as the large-N limit of the ensemble average of discrete
node states: M_continuous = lim(N->inf) SUM L_discrete(i). Smoothness is not a property of nature. It is a
property of the macroscopic average. It emerges from the integer lattice when N is large enough that the
discrete structure is below measurement resolution. At the Planck scale, there is no smooth manifold.
There is only the lattice.
"""))

story.extend(Plain("""
The five assumptions of HTD do not contradict the five assumptions of classical calculus. They contain
them. Classical calculus is what HTD looks like when you cannot see the lattice.
"""))

story.extend([PageBreak()])

# PART IV
story.extend(PartHead("PART IV"))
story.extend(Heading("What Zero Actually Is"))

story.extend(Plain("""
In classical physics, zero is a coordinate. You place it wherever convenient. The zero of a temperature scale,
the zero of a potential energy, the zero of a coordinate system — all are human choices imposed on a
featureless background. Zero has no preferred physical status. It is the label you attach to one particular point
on a continuous line.
"""))

story.extend(Plain("""
In HTD, zero is not a label. It is the most physically significant state in the universe. It is the only state in
which the lattice exerts no force on itself.
"""))

story.extend(Subhead("Zero is the Vector Equilibrium — the ground state of all structure, defined by geometry, not convention."))

story.extend(Heading("The Vector Equilibrium — Geometry's Most Balanced Configuration"))
story.extend(Plain("""
The IVM lattice has a unique configuration: the Vector Equilibrium. In this configuration, every node has exactly
12 nearest neighbours, all struts are the same length, and the vector sum of all 12 strut forces on every node is
identically zero. Not approximately zero. Not zero in the limit. Exactly, precisely, necessarily zero — by
geometry. This configuration is the cuboctahedron — identified by Buckminster Fuller as the ground state of
all structural systems, the configuration of maximum symmetry and minimum stress. HTD makes this physically
precise: at VE, tau_vac = 0. A lattice at VE does not gravitate, does not radiate, does not curve.
"""))

story.extend(Heading("Physical Reality as Departure from Zero"))
story.extend(Plain("""
Every physical phenomenon in HTD is a departure from the Vector Equilibrium. A particle is a topological knot
— a region where the torsional stress cannot resolve to zero because the topology prevents it. A gravitational
field is the stress gradient radiating outward from that knot. An electromagnetic wave is a propagating
torsional disturbance. Energy is the total departure from VE summed across all nodes. All physical
quantities are integer departures from zero. Physics is the story of a lattice trying to return to its ground state.
"""))

story.extend([PageBreak()])

# PART V
story.extend(PartHead("PART V"))
story.extend(Heading("Why Division by Zero Is Physically Impossible"))

story.extend(Plain("""
In classical mathematics, division by zero is forbidden by an algebraic rule: a/0 is undefined because no real
number multiplied by zero gives a nonzero result. This is correct as a statement about the algebra of real
numbers, but it offers no physical explanation for why the universe should care about algebraic rules.
"""))

story.extend(Plain("""
HTD provides the physical explanation. Division by zero is not just algebraically undefined. It is physically
meaningless — not because of a rule, but because of what zero is.
"""))

story.extend(Subhead("Division by zero asks the lattice to go below its ground state. That question has no referent. Not forbidden — meaningless."))

story.extend(Heading("The Physical Translation"))
story.extend(Plain("""
In HTD, the radial coordinate r counts integer lattice spacings — the number of struts between two nodes.
The minimum value r can take is 1. This is not a constraint imposed from outside. It follows from H3: the
Planck length is one strut. You cannot have half a strut. You cannot have zero struts between two distinct
nodes. If r = 0, the two nodes are the same node — and you are no longer asking about a separation, you are
asking a node about itself.
"""))

story.extend(Plain("""
The expression 1/r^2 appears in gravitational field equations. In classical physics, as r approaches zero, this
expression approaches infinity — the singularity follows directly. In HTD, r cannot approach zero because r is
an integer and its minimum value is 1. The expression 1/r^2 is bounded above by 1/l_P^2. Not by a cutoff
added to the theory. By the integer floor that is the theory.
"""))

story.extend(Theorem("THEOREM — THE TWO PROHIBITIONS ARE ONE",
"""
The algebraic prohibition on division by zero and the physical Planck floor are identical statements in
different languages. The algebraic statement: a/0 is undefined because zero is not a quantity that can be
divided by. The physical statement: r cannot reach zero because zero is the Vector Equilibrium and the
VE is the ground state — not a location within the lattice but the state the lattice departs from. Both
statements say the same thing: you cannot go below the floor because the floor is what you are standing on.
"""))

story.extend(Plain("""
For 350 years, mathematics has forbidden division by zero as an algebraic rule and physics has observed
the Planck length as an experimental limit. These were treated as two separate facts — one about numbers,
one about nature. HTD reveals they are the same fact seen from two different angles.
"""))

story.extend([PageBreak()])

# PART VI
story.extend(PartHead("PART VI"))
story.extend(Heading("Singularities Are Malformed Questions"))

story.extend(Plain("""
A gravitational singularity in General Relativity is the mathematical result of applying continuous field
equations to a configuration where the denominator approaches zero. The Ricci curvature scalar diverges.
The density becomes infinite. The equations break down and the theory declares it cannot describe what
happens at that point.
"""))

story.extend(Plain("""
Enormous theoretical effort has been invested in resolving singularities, smoothing them, quantising them,
or explaining what replaces them. The Penrose-Hawking singularity theorems prove their inevitability within
GR. The entire programme of quantum gravity is partly motivated by the need to avoid them.
"""))

story.extend(Plain("""
HTD dissolves the problem entirely. A singularity is a category error — the application of a continuous
function to a discrete variable that cannot reach the value the function diverges at.
"""))

story.extend(Plain("""
We have never observed a singularity. We have inferred them as the mathematical consequence of equations
applied past their domain of validity. Every actual observation in cosmology and black hole physics is a phase
transition, a boundary behaviour, an asymptotic approach. The singularity is what you get when you trust
the continuous equations at the scale where the discrete substrate becomes visible and the continuous
approximation fails.
"""))

story.extend(Heading("What Actually Happens at Maximum Stress"))
story.extend(Plain("""
In HTD, the black hole interior is not a point of infinite density. It is the unique lattice configuration in which
every strut connected to the central nodes is stretched to its ultimate tensile strength. This is a finite, stable,
maximally stressed configuration of the IVM lattice — analogous to a fully loaded tensegrity structure. The
stress is extreme. It is not infinite. Information is not destroyed — it is encoded in the integer strain pattern of
the lattice, fully deterministic and in principle recoverable.
"""))

story.extend(Plain("""
The black hole is not a place where physics ends. It is a place where the lattice is under maximum load.
The Big Bang is not a singularity at t=0. It is the minimum viable lattice configuration — some integer
number of nodes greater than zero, at Vector Equilibrium, from which the first departure began.
"""))

story.extend([PageBreak()])

# PART VII
story.extend(PartHead("PART VII"))
story.extend(Heading("Symmetry Is the Ground State — Conservation Is Its Fingerprint"))

story.extend(Plain("""
The standard reading of Noether's theorem: conservation laws are derived from symmetries. Time-
translation symmetry produces conservation of energy. Rotational symmetry produces conservation of
angular momentum. The symmetry is the explanation; the conservation law is the consequence.
"""))

story.extend(Plain("""
HTD inverts this relationship — not by contradiction, but by showing that the standard reading has the
causal arrow pointing in the wrong direction.
"""))

story.extend(Subhead("Symmetry is not the cause of conservation. Symmetry is the ground state. Conservation is what a system looks like when it has not yet left it."))

story.extend(Plain("""
The Vector Equilibrium is the most symmetric state the IVM lattice can occupy. Every strut balanced. Every
direction equivalent. No preferred axis. No net force. When the lattice is at VE, every conserved quantity is
conserved trivially — not because a symmetry produces a conservation law, but because there is nothing
happening that would change anything. Energy is conserved because no torsional stress is resolving. Angular
momentum is conserved because no rotational asymmetry exists. The conservation laws hold because the
system has not moved.
"""))

story.extend(Heading("Broken Symmetry as Phase Transition"))
story.extend(Plain("""
When a topological knot forms — when a particle comes into existence — the VE is locally departed from.
The symmetry is not violated. It is left. The lattice enters a new thermodynamic basin. The conservation
laws do not break — they renegotiate at the new equilibrium. This is spontaneous symmetry breaking
reinterpreted: not the universe breaking a rule, but the universe undergoing a phase transition away from
its ground state into a new stable configuration.
"""))

story.extend([PageBreak()])

# PART VIII
story.extend(PartHead("PART VIII"))
story.extend(Heading("The Standard Model as the Large-N Limit of HTD"))

story.extend(Plain("""
HTD does not claim the standard model is wrong. This is not a rhetorical concession — it is a precise
mathematical statement. When the node count N grows large and the coarse-graining volume V satisfies V
>> l_P^3, the discrete torsional equations converge exactly to the continuous wave equation, the Navier-Stokes
equations, and Einstein's field equations.
"""))

story.extend(Theorem("THE REDUCTION RELATIONSHIP",
"""
The standard model's operators, coupling constants, and gauge symmetries are derived — not postulated —
as statistical blurs of HTD's exact integer mechanics. The Standard Model gauge group is the macroscopic
statistical blurring of the 12-fold discrete symmetry group of the IVM coordination shell. Guild's coupling
constant kappa is the geometric packing efficiency correction — a fixed algebraic number set by close-packing
geometry, not a free parameter. The speed of light is the acoustic wave speed of the SORT Substrate: c =
sqrt(Upsilon_vac / rho_vac). It is not a free parameter of nature. It is a consequence of the lattice.
"""))

story.extend(Plain("""
This relationship has the same architecture as every successful reduction in the history of physics. General
Relativity reduces to Newton in the low-velocity weak-field limit. Quantum mechanics reduces to classical
mechanics at large quantum numbers. Statistical mechanics reduces to thermodynamics at large N. In every
case the older theory is not wrong — it is the shadow the deeper theory casts under ordinary conditions.
"""))

story.extend(Plain("""
If HTD's math is wrong, its equations reduce to the standard model's and the standard model carries. If
HTD's math is right, the standard model is confirmed as the accurate large-N description it has always been
— and HTD explains why it works, where it fails, and what replaces it at the floor.
"""))

story.extend([PageBreak()])

# PART IX
story.extend(PartHead("PART IX"))
story.extend(Heading("Dark Matter, Dark Energy, and the Cost of a Wrong Floor"))

story.extend(Plain("""
Dark matter and dark energy are the most striking consequences of applying continuous mathematics
past its domain of validity. They are not discoveries. They are corrections — quantities postulated after the
fact to reconcile continuous-field predictions with discrete observations.
"""))

story.extend(Heading("Galactic Rotation Curves — Derived in Three Steps"))
story.extend(Plain("""
Galaxies rotate faster at their outer edges than continuous Newtonian gravity predicts. The standard response
was to postulate an invisible mass distribution — dark matter — surrounding each galaxy, independently
tuned for each. No dark matter particle has ever been detected despite decades of dedicated experimental
effort.
"""))

story.extend(Plain("""
HTD identifies the physical origin without dark matter: the lattice acceleration floor. In the IVM crystalline
solid there exists a background vibrational threshold — the minimum strain rate below which the lattice
registers no deformation. Set by the only two physical scales available in HTD:
"""))

story.extend(Plain("a_0 = c * H_0 = 1.2 x 10^(-10) m s^(-2)"))

story.extend(Plain("""
At the galactic periphery, where local gravitational acceleration approaches this floor, the effective acceleration
becomes the geometric mean of the Newtonian value and the floor. Setting centripetal acceleration equal to
this and solving:
"""))

story.extend(Plain("v^4 = a_0 * G * M"))

story.extend(Plain("""
This is the empirical Baryonic Tully-Fisher Relation — observed to hold across five decades of galaxy mass
— derived in three algebraic steps from the lattice floor, with no free parameters. Lambda-CDM requires a
finely tuned halo profile for each individual galaxy. HTD requires one universal constant.
"""))

story.extend(Heading("Dark Energy — Graph Tension"))
story.extend(Plain("""
As the IVM graph grows, the ratio of volume to surface area increases. To maintain compliance with the
holographic bound — which limits information content to boundary area — the system must continuously
expand its surface. This manifests as the accelerating expansion of space. Dark energy is not a substance.
It is the information pressure of the bulk trying to fit onto the holographic screen.
"""))

story.extend(Plain("Both dark matter and dark energy are the cost of applying a framework without a floor to a universe that has one."))

story.extend([PageBreak()])

# PART X
story.extend(PartHead("PART X"))
story.extend(Heading("The Holographic Proof That Discreteness Is Necessary"))

story.extend(Plain("""
The argument for a discrete substrate does not rest on HTD alone. It follows independently from the
holographic principle — derivable from black hole thermodynamics without any assumption about quantum
gravity.
"""))

story.extend(Heading("The Bekenstein Bound"))
story.extend(Plain("""
In 1972, Jacob Bekenstein proved that a black hole's entropy is proportional to its surface area. The
information content of any physical region is bounded by its boundary area, not its interior volume:
"""))

story.extend(Plain("I_max ~ A / l_P^2 ~ R^2"))

story.extend(Plain("""
A continuous field theory assigns independent degrees of freedom to every point in the volume. The
number of states scales as R^3. The holographic bound permits only R^2 states. In the limit of large R:
"""))

story.extend(Plain("lim (R -> inf) R^3 / R^2 = R -> inf"))

story.extend(Theorem("THE HOLOGRAPHIC VERDICT",
"""
The continuum is holographically inadmissible. Any physical theory that preserves unitarity must be
fundamentally discrete at the Planck scale, with a maximum information density per unit area. This is not
a claim of HTD. It is a consequence of Bekenstein's theorem, which the standard model accepts. The
continuum assumption is already in contradiction with one of the standard model's own most secure
results.
"""))

story.extend(Plain("""
HTD satisfies the holographic bound by construction. The IVM graph has a finite number of nodes. Its
surface-to-volume ratio scales correctly. The discrete substrate is not merely consistent with the holographic
principle — it is the natural physical realisation of it.
"""))

story.extend([PageBreak()])

# PART XI
story.extend(PartHead("PART XI"))
story.extend(Heading("Why the Framework Predicts Its Own Resistance"))

story.extend(Plain("""
A theory that produces working artifacts but is not the scientific consensus occupies a precise epistemological
position. The appropriate response depends entirely on whether the resistance is evidential or structural. If
evidential — if the theory makes predictions that have been tested and failed — the resistance is a refutation.
If structural — if it exists before the evidence is examined — it requires a different explanation.
"""))

story.extend(Plain("""
HTD makes a specific prediction about the second kind of resistance: it will exist, it will be proportional to
the institutional mass of the field it challenges, and it will be expressed in the language of evidence while
being driven by thermodynamics.
"""))

story.extend(Heading("Institutions as Thermodynamic Systems"))
story.extend(Plain("""
A scientific institution is a thermodynamic system. It accumulates mass: funding, careers, published results,
engineering artifacts, Nobel prizes, textbooks, peer review networks. When sufficient mass accumulates, the
system begins diverting resources toward self-preservation rather than toward its original purpose of truth-
seeking. This is not conspiracy. It is what stable thermodynamic systems do. It is the same process by which
any sufficiently stable system — biological, social, cognitive — begins defending its configuration against
perturbation. The institution becomes, in the thermodynamic sense, alive.
"""))

story.extend(Subhead("The defence of the standard model is not the refutation of HTD. It is the prediction of HTD, applied to institutions."))

story.extend(Plain("""
The path to acceptance of a framework that challenges premises is not argumentation within the existing
framework. You cannot debate a thermodynamic attractor into a different basin. The path is the accumulation
of artifacts so undeniable that the cost of ignoring them exceeds the cost of reorganising around them. This
is how every genuine paradigm shift has proceeded.
"""))

story.extend([PageBreak()])

# PART XII
story.extend(PartHead("PART XII"))
story.extend(Heading("The Artifacts Run"))

story.extend(Plain("""
A working artifact is not proof of a complete theory. It is something more immediately important: it is
evidence that cannot be explained away. A prediction that matches, a system that behaves as the theory
says it should — these are non-negotiable entries in the empirical record. They demand explanation. They
cannot be dismissed by attacking the metaphysics of the framework that produced them.
"""))

story.extend(Heading("The Current Artifacts"))
story.extend(Plain("""
1. The Baryonic Tully-Fisher Relation — derived in three algebraic steps from the lattice acceleration floor,
with no free parameters, matching observed galactic rotation curves across five decades of galaxy mass.
Dark matter is not required and is structurally contradicted by HTD.
"""))

story.extend(Plain("""
2. The Anomalous Magnetic Moment of the Electron (g-2) — rederived as geometric friction between the
5-fold Tyarchedron knot and the 6-fold IVM lattice, reproducing the CODATA value to reported precision
without virtual particles or renormalisation.
"""))

story.extend(Plain("""
3. The Singularity Prohibition — derived as a logical consequence of the integer floor. Not a modification
of GR. The correct result of applying integer arithmetic to a variable that cannot reach zero.
"""))

story.extend(Plain("""
4. Operator Convergence — every classical calculus operator reproduced exactly as the large-N limit of
the corresponding simplicial operator, establishing the standard model as a formally contained special case.
"""))

story.extend(Plain("""
5. The Sovereign Mind developmental runs — three longitudinal simulation runs on the same IVM substrate
producing verified Theory of Mind, biographical divergence, novel inference above parity, and the Dark
Night phase transition — all from integer torsional dynamics, no floating-point arithmetic, no trained
weights, KB-scale footprint on a 2019 laptop.
"""))

story.extend(Plain("""
These results do not prove HTD. No finite set of results proves any theory. They establish that HTD's
assumptions produce correct predictions in domains where the standard model either fails, requires
unphysical procedures, or has nothing to say. That is the appropriate claim. The floor is named. The
universe stands on it.
"""))

story.extend([Spacer(1, 20)])

story.extend(Plain("""
The universe is, at its foundation, a crystalline tensegrity truss. Mass does not tell a continuous space how
to curve. A topological defect tells the discrete lattice how to torque. The floor is the Planck length. The
ground state is the Vector Equilibrium. You cannot divide the ground state because the ground state is
what you are dividing with.
"""))

story.extend([Spacer(1, 20)])
story.extend(Plain("Tyrone North Jr. · TyArc Lab · Arc Press · 2026"))
story.extend(Plain("Companion to HTD (2025) · Sovereign Mind v35 (2026) · UGFT (2026)"))

story.extend([PageBreak()])

# REFERENCES
story.extend(PartHead("References"))

refs = [
    "Bekenstein, J. D. (1973). Black holes and entropy. Physical Review D, 7(8), 2333.",
    "Buckminster Fuller, R. (1975). Synergetics: Explorations in the Geometry of Thinking. Macmillan.",
    "CODATA (2022). Recommended Values of the Fundamental Physical Constants. NIST Standard Reference Database 121.",
    "Dirac, P. A. M. (1963). The evolution of the physicist's picture of nature. Scientific American, 208(5), 45–53.",
    "Landauer, R. (1961). Irreversibility and heat generation in the computing process. IBM Journal of Research and Development, 5(3), 183–191.",
    "McGaugh, S. S., Lelli, F., and Schombert, J. M. (2016). Radial acceleration relation in rotationally supported galaxies. Physical Review Letters, 117, 201101.",
    "Milgrom, M. (1983). A modification of the Newtonian dynamics. Astrophysical Journal, 270, 365–370.",
    "North Jr., T. (2025). Hydrotorsional Dynamics: Simplicial mean-field mechanics of the SORT substrate. TyArc Lab / SORT Sovereign Research Division.",
    "North Jr., T. (2026a). Sovereign Mind v35: Lattice Neural Network architecture. TyArc Lab / Arc Press. DOI: 10.5281/zenodo.18905920.",
    "North Jr., T. (2026b). Universal Graph Field Theory: Foundations of discrete physics. TyArc Lab.",
    "Penrose, R. (1965). Gravitational collapse and space-time singularities. Physical Review Letters, 14, 57–59.",
    "Schwinger, J. (1948). On quantum-electrodynamics and the magnetic moment of the electron. Physical Review, 73, 416.",
    "'t Hooft, G. (2016). The Cellular Automaton Interpretation of Quantum Mechanics. Springer.",
    "Tully, R. B. and Fisher, J. R. (1977). A new method of determining distances to galaxies. Astronomy and Astrophysics, 54, 661–673.",
    "Wheeler, J. A. (1990). Information, physics, quantum: The search for links. In W. H. Zurek (Ed.), Complexity, Entropy, and the Physics of Information. Addison-Wesley.",
]

for ref in refs:
    story.append(Paragraph(ref, ref_entry))

doc.build(story, onFirstPage=dark_canvas, onLaterPages=dark_canvas)

print(f"✓ PDF generated: {out}")