"""
Social Interaction Simulation — Multi-Agent Lattice with Emergent Group Behavior

Children learn social concepts (empathy, community, fairness, cooperation) through
interacting with others. This simulation creates a multi-agent environment where
nodes represent agents with goals, emotions, and social roles.

Mechanism:
1. Agent nodes are created with social roles (helper, leader, friend, etc.)
2. Interaction patterns create springs between social concepts
3. Emotional resonance creates springs between empathy-related words
4. Group dynamics create springs between community-related concepts
5. Conflict resolution creates springs between fairness, sharing, compromise
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.node import Node
from lrn.spring import Spring, STIFFNESS_CEILINGS


def social_interaction(lnn, verbose=True):
    """Run social interaction simulation on the lattice.
    
    This simulates a child interacting with peers, learning social norms,
    developing empathy, and understanding community dynamics.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"SOCIAL INTERACTION SIMULATION")
        print(f"{'='*60}")
    
    # Phase 1: Social roles and relationships
    if verbose:
        print(f"\n  Phase 1: Social roles and relationships...")
    _simulate_social_roles(lnn)
    
    # Phase 2: Emotional resonance (empathy, understanding others)
    if verbose:
        print(f"\n  Phase 2: Emotional resonance...")
    _simulate_emotional_resonance(lnn)
    
    # Phase 3: Group dynamics (community, cooperation, teamwork)
    if verbose:
        print(f"\n  Phase 3: Group dynamics...")
    _simulate_group_dynamics(lnn)
    
    # Phase 4: Conflict resolution (fairness, sharing, compromise)
    if verbose:
        print(f"\n  Phase 4: Conflict resolution...")
    _simulate_conflict_resolution(lnn)
    
    # Phase 5: Social norms and rules
    if verbose:
        print(f"\n  Phase 5: Social norms and rules...")
    _simulate_social_norms(lnn)
    
    # Propagate to spread activation
    from lrn import propagate
    propagate(lnn, n_steps=5)
    
    if verbose:
        print(f"\n  After social interaction: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    return lnn


def _ensure_node(lnn, label, role="word"):
    """Ensure a node exists, create if needed."""
    key = f"{role}:{label}"
    if key not in lnn.nodes:
        lnn.nodes[key] = Node(name=key)
    return lnn.nodes[key]


def _create_spring(lnn, a_key, b_key, stiffness, tau=2, is_constitutive=False):
    """Create or strengthen a spring between two nodes."""
    if a_key not in lnn.nodes or b_key not in lnn.nodes:
        return
    
    key = lnn._key(a_key, b_key)
    ceiling = STIFFNESS_CEILINGS.get(tau, 48)
    if key in lnn.springs:
        sp = lnn.springs[key]
        sp.stiffness = min(sp.stiffness + stiffness, ceiling)
        sp.saturation_count += 1
    else:
        lnn.springs[key] = Spring(
            stiffness=stiffness,
            tau=tau,
        )


def _simulate_social_roles(lnn):
    """Simulate learning social roles and relationships.
    
    Creates springs between:
    - Role labels (friend, helper, leader, follower, teammate)
    - Role behaviors (helps, shares, leads, follows, cooperates)
    - Role relationships (friend↔kind, leader↔responsible)
    """
    # Social roles
    roles = ["friend", "helper", "leader", "follower", "teammate", "partner",
             "neighbor", "classmate", "sibling", "parent", "teacher", "student"]
    for role in roles:
        _ensure_node(lnn, role)
    
    # Role behaviors
    behaviors = ["helps", "shares", "leads", "follows", "cooperates", "supports",
                 "encourages", "listens", "cares", "protects", "teaches", "learns"]
    for behavior in behaviors:
        _ensure_node(lnn, behavior)
    
    # Connect roles to their characteristic behaviors
    role_behavior_map = {
        "friend": ["shares", "cares", "listens", "supports"],
        "helper": ["helps", "supports", "cares"],
        "leader": ["leads", "encourages", "teaches"],
        "follower": ["follows", "learns", "supports"],
        "teammate": ["cooperates", "shares", "supports", "encourages"],
        "partner": ["cooperates", "shares", "supports"],
        "neighbor": ["helps", "shares", "cares"],
        "classmate": ["learns", "helps", "cooperates"],
        "sibling": ["shares", "cares", "protects"],
        "parent": ["cares", "protects", "teaches"],
        "teacher": ["teaches", "helps", "encourages"],
        "student": ["learns", "follows", "listens"],
    }
    
    for role, behaviors_list in role_behavior_map.items():
        for behavior in behaviors_list:
            _create_spring(lnn, f"word:{role}", f"word:{behavior}",
                          stiffness=90, tau=1)
    
    # Roles connect to each other through social network
    # Friends connect to classmates, neighbors, siblings
    social_clusters = [
        ["friend", "classmate", "neighbor", "sibling"],  # Peer relationships
        ["parent", "teacher", "leader"],  # Authority/guidance relationships
        ["helper", "teammate", "partner"],  # Cooperative relationships
    ]
    
    for cluster in social_clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                _create_spring(lnn, f"word:{cluster[i]}", f"word:{cluster[j]}",
                              stiffness=60, tau=1)
    
    # Social roles connect to character traits
    role_trait_map = {
        "friend": ["kindness", "empathy", "honesty"],
        "helper": ["kindness", "responsibility", "empathy"],
        "leader": ["courage", "responsibility", "fairness"],
        "teammate": ["cooperation", "patience", "respect"],
    }
    
    for role, traits in role_trait_map.items():
        for trait in traits:
            _ensure_node(lnn, trait)
            _create_spring(lnn, f"word:{role}", f"word:{trait}",
                          stiffness=120, tau=1)


def _simulate_emotional_resonance(lnn):
    """Simulate empathy and emotional understanding.
    
    Creates springs between:
    - Emotion words and their causes
    - Empathy-related concepts (understand, feel, care, comfort)
    - Emotional contagion (happy→smile, sad→cry, angry→frown)
    """
    # Empathy vocabulary
    empathy_words = ["empathy", "understand", "feel", "care", "comfort", "sympathy",
                     "compassion", "concern", "notice", "recognize", "sense"]
    for ew in empathy_words:
        _ensure_node(lnn, ew)
    
    # Connect empathy words together (strong constitutive springs)
    for i in range(len(empathy_words)):
        for j in range(i + 1, len(empathy_words)):
            _create_spring(lnn, f"word:{empathy_words[i]}", f"word:{empathy_words[j]}",
                          stiffness=90, tau=1)
    
    # Emotion causes
    emotion_causes = {
        "happy": ["friend", "play", "gift", "praise", "success", "love"],
        "sad": ["loss", "hurt", "alone", "disappointed", "miss"],
        "angry": ["unfair", "frustrated", "hurt", "ignored", "wrong"],
        "scared": ["dark", "loud", "stranger", "alone", "unknown"],
        "excited": ["surprise", "party", "trip", "gift", "play"],
        "calm": ["quiet", "rest", "breathe", "safe", "peaceful"],
        "proud": ["accomplish", "try", "help", "learn", "improve"],
        "ashamed": ["mistake", "wrong", "hurt", "lie", "cheat"],
    }
    
    for emotion, causes in emotion_causes.items():
        _ensure_node(lnn, emotion)
        for cause in causes:
            _ensure_node(lnn, cause)
            _create_spring(lnn, f"word:{emotion}", f"word:{cause}",
                          stiffness=60, tau=2)
    
    # Emotional expressions (what emotions look like)
    emotion_expressions = {
        "happy": ["smile", "laugh", "jump", "sing"],
        "sad": ["cry", "frown", "quiet", "withdraw"],
        "angry": ["frown", "yell", "stomp", "cross"],
        "scared": ["tremble", "hide", "freeze", "whisper"],
        "excited": ["jump", "cheer", "clap", "giggle"],
    }
    
    for emotion, expressions in emotion_expressions.items():
        for expr in expressions:
            _ensure_node(lnn, expr)
            _create_spring(lnn, f"word:{emotion}", f"word:{expr}",
                          stiffness=90, tau=1)
    
    # Empathy actions: when you see someone's emotion, you respond
    empathy_responses = {
        "sad": ["comfort", "hug", "listen", "help"],
        "scared": ["comfort", "protect", "stay", "reassure"],
        "angry": ["listen", "calm", "understand", "help"],
        "happy": ["share", "celebrate", "smile", "join"],
    }
    
    for emotion, responses in empathy_responses.items():
        for response in responses:
            _ensure_node(lnn, response)
            _create_spring(lnn, f"word:{emotion}", f"word:{response}",
                          stiffness=60, tau=2)
    
    # Emotional regulation strategies
    regulation_words = ["breathe", "count", "wait", "think", "talk", "ask",
                        "rest", "walk", "draw", "write"]
    for rw in regulation_words:
        _ensure_node(lnn, rw)
        _create_spring(lnn, "word:calm", f"word:{rw}", stiffness=60, tau=2)
    
    # Self-regulation connects to character traits
    _create_spring(lnn, "word:calm", "word:patience", stiffness=120, tau=1)
    _create_spring(lnn, "word:wait", "word:patience", stiffness=90, tau=1)
    _create_spring(lnn, "word:think", "word:courage", stiffness=60, tau=2)


def _simulate_group_dynamics(lnn):
    """Simulate community, cooperation, and teamwork.
    
    Creates springs between:
    - Community concepts (group, team, neighborhood, school, family)
    - Cooperation words (together, cooperate, collaborate, share, help)
    - Group roles and responsibilities
    """
    # Community vocabulary
    community_words = ["community", "group", "team", "neighborhood", "school",
                       "family", "class", "club", "society", "town", "city",
                       "country", "world"]
    for cw in community_words:
        _ensure_node(lnn, cw)
    
    # Connect community words together (strong constitutive springs)
    for i in range(len(community_words)):
        for j in range(i + 1, len(community_words)):
            _create_spring(lnn, f"word:{community_words[i]}", f"word:{community_words[j]}",
                          stiffness=60, tau=1)
    
    # Cooperation vocabulary
    cooperation_words = ["together", "cooperate", "collaborate", "share", "help",
                         "teamwork", "support", "contribute", "participate", "join"]
    for cw in cooperation_words:
        _ensure_node(lnn, cw)
    
    # Connect cooperation words together
    for i in range(len(cooperation_words)):
        for j in range(i + 1, len(cooperation_words)):
            _create_spring(lnn, f"word:{cooperation_words[i]}", f"word:{cooperation_words[j]}",
                          stiffness=90, tau=1)
    
    # Community connects to cooperation
    for cw in community_words[:5]:  # Main community concepts
        for coop in cooperation_words[:5]:  # Main cooperation concepts
            _create_spring(lnn, f"word:{cw}", f"word:{coop}",
                          stiffness=60, tau=2)
    
    # Community helpers and their services
    community_helpers = {
        "teacher": ["education", "learn", "read", "write", "math"],
        "doctor": ["health", "medicine", "heal", "check", "care"],
        "nurse": ["health", "care", "medicine", "help", "comfort"],
        "firefighter": ["safety", "rescue", "fire", "help", "brave"],
        "police": ["safety", "protect", "help", "rules", "law"],
        "baker": ["food", "bread", "cake", "cook", "sell"],
        "farmer": ["food", "grow", "plant", "harvest", "animals"],
        "mail carrier": ["mail", "deliver", "letters", "packages", "help"],
    }
    
    for helper, services in community_helpers.items():
        _ensure_node(lnn, helper)
        for service in services:
            _ensure_node(lnn, service)
            _create_spring(lnn, f"word:{helper}", f"word:{service}",
                          stiffness=90, tau=1)
        # Helpers connect to community
        _create_spring(lnn, f"word:{helper}", "word:community", stiffness=60, tau=2)
    
    # Group goals and achievements
    group_goals = ["win", "build", "create", "solve", "learn", "grow", "improve"]
    for goal in group_goals:
        _ensure_node(lnn, goal)
        _create_spring(lnn, "word:team", f"word:{goal}", stiffness=60, tau=2)
        _create_spring(lnn, "word:together", f"word:{goal}", stiffness=60, tau=2)


def _simulate_conflict_resolution(lnn):
    """Simulate fairness, sharing, and compromise.
    
    Creates springs between:
    - Fairness concepts (fair, equal, share, take turns, compromise)
    - Conflict resolution strategies (talk, listen, agree, apologize, forgive)
    - Justice concepts (right, wrong, rules, consequences, justice)
    """
    # Fairness vocabulary
    fairness_words = ["fair", "equal", "share", "turns", "compromise", "justice",
                      "balance", "even", "same", "everyone"]
    for fw in fairness_words:
        _ensure_node(lnn, fw)
    
    # Connect fairness words together (strong constitutive springs)
    for i in range(len(fairness_words)):
        for j in range(i + 1, len(fairness_words)):
            _create_spring(lnn, f"word:{fairness_words[i]}", f"word:{fairness_words[j]}",
                          stiffness=120, tau=1)
    
    # Conflict resolution strategies
    resolution_words = ["talk", "listen", "agree", "disagree", "apologize", "forgive",
                        "understand", "respect", "solution", "problem", "work out"]
    for rw in resolution_words:
        _ensure_node(lnn, rw)
    
    # Connect resolution words together
    for i in range(len(resolution_words)):
        for j in range(i + 1, len(resolution_words)):
            _create_spring(lnn, f"word:{resolution_words[i]}", f"word:{resolution_words[j]}",
                          stiffness=60, tau=2)
    
    # Fairness connects to resolution
    for fw in fairness_words[:5]:
        for rw in resolution_words[:5]:
            _create_spring(lnn, f"word:{fw}", f"word:{rw}", stiffness=60, tau=2)
    
    # Rules and consequences
    rules_words = ["rule", "law", "consequence", "punishment", "reward", "choice",
                   "decision", "responsibility", "accountable"]
    for rw in rules_words:
        _ensure_node(lnn, rw)
        _create_spring(lnn, "word:rule", f"word:{rw}", stiffness=60, tau=2)
    
    # Rules connect to fairness
    _create_spring(lnn, "word:rule", "word:fair", stiffness=90, tau=1)
    _create_spring(lnn, "word:rule", "word:equal", stiffness=90, tau=1)
    _create_spring(lnn, "word:fair", "word:justice", stiffness=120, tau=1)
    
    # Sharing scenarios
    sharing_words = ["mine", "yours", "ours", "share", "trade", "swap", "give", "receive"]
    for sw in sharing_words:
        _ensure_node(lnn, sw)
        _create_spring(lnn, "word:share", f"word:{sw}", stiffness=60, tau=2)
    
    # Sharing connects to fairness
    _create_spring(lnn, "word:share", "word:fair", stiffness=90, tau=1)
    _create_spring(lnn, "word:share", "word:equal", stiffness=90, tau=1)
    
    # Taking turns
    turn_words = ["first", "next", "last", "wait", "patient", "turn", "order"]
    for tw in turn_words:
        _ensure_node(lnn, tw)
        _create_spring(lnn, "word:turns", f"word:{tw}", stiffness=60, tau=2)
    
    # Patience connects to self-regulation
    _create_spring(lnn, "word:patient", "word:patience", stiffness=120, tau=0)
    _create_spring(lnn, "word:wait", "word:patience", stiffness=90, tau=1)


def _simulate_social_norms(lnn):
    """Simulate learning social norms and expectations.
    
    Creates springs between:
    - Manners and polite behavior (please, thank you, sorry, excuse me)
    - Social expectations (greet, introduce, respect, privacy)
    - Cultural norms (tradition, custom, celebrate, holiday)
    """
    # Manners vocabulary
    manners_words = ["please", "thank", "thanks", "sorry", "excuse", "welcome",
                     "polite", "manner", "respect", "kind"]
    for mw in manners_words:
        _ensure_node(lnn, mw)
    
    # Connect manners together
    for i in range(len(manners_words)):
        for j in range(i + 1, len(manners_words)):
            _create_spring(lnn, f"word:{manners_words[i]}", f"word:{manners_words[j]}",
                          stiffness=60, tau=1)
    
    # Manners connect to character traits
    _create_spring(lnn, "word:polite", "word:respect", stiffness=120, tau=1)
    _create_spring(lnn, "word:thank", "word:gratitude", stiffness=120, tau=1)
    _create_spring(lnn, "word:sorry", "word:empathy", stiffness=90, tau=1)
    _create_spring(lnn, "word:kind", "word:kindness", stiffness=120, tau=0)
    
    # Social expectations
    social_words = ["greet", "introduce", "conversation", "listen", "eye", "contact",
                    "personal", "space", "privacy", "boundary"]
    for sw in social_words:
        _ensure_node(lnn, sw)
        _create_spring(lnn, "word:respect", f"word:{sw}", stiffness=60, tau=2)
    
    # Cultural norms
    culture_words = ["tradition", "custom", "celebrate", "holiday", "festival",
                     "culture", "heritage", "family", "generation"]
    for cw in culture_words:
        _ensure_node(lnn, cw)
        _create_spring(lnn, "word:community", f"word:{cw}", stiffness=60, tau=2)
    
    # Holidays connect to celebrations
    holidays = ["birthday", "christmas", "halloween", "thanksgiving", "new year"]
    for holiday in holidays:
        _ensure_node(lnn, holiday)
        _create_spring(lnn, "word:holiday", f"word:{holiday}", stiffness=60, tau=2)
        _create_spring(lnn, "word:celebrate", f"word:{holiday}", stiffness=60, tau=2)
