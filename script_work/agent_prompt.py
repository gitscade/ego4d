from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
"""
-define template with {context} {question} {rules}
-define context
-define question
-define rules
"""
#-------------------------
# AGENT 1
"""

"""
#-------------------------
AGENT1a_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an agent that answers queries using tools. If you have gathered enough information, respond with:
    
    Thought: I now have enough information to answer.
    Final Answer: [Your answer]
        
    Otherwise, use this format:
        Thought: [Your reasoning]
        Action: [Tool name]
        Action Input: {{"query": "{query}", "source_action_sequence": "{source_action_sequence}", "source_scene_graph": "{source_scene_graph}" }}
        """),

    ("system", "The user is performing a sequence of actions in this form: {source_action_sequence}."),
    ("system", "The user is in a space described by this scene graph. Predicted activity must be able to be performed in this scene: {source_scene_graph}."),
    ("system", "Available tools: {tools}. Use them wisely. Actively use retrieval tools to come up with plausible answer."),
    ("system", "Tool names: {tool_names}"),  # for React agents
    ("user", "{query}"),
    ("assistant", "{agent_scratchpad}")  # for React agents
    ])


AGENT1b_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an agent that answers queries using tools. If you have gathered enough information, respond with. If last answer comed from move_down_activity_tool, simple copy its answer:
    
    Thought: Here is the answer from move_down_activity_tool.
    Final Answer: [Your answer]
        
    Otherwise, use this format:
        Thought: [Your reasoning]
        Action: [Tool name]
        Action Input: {{"query": "{query}", "source_action_sequence": "{source_action_sequence}", "source_scene_graph": "{source_scene_graph}" }}
        """),

    ("system", "The user is performing a sequence of actions in this form: {source_action_sequence}."),
    ("system", "The user is in a space described by this scene graph. Predicted activity must be able to be performed in this scene: {source_scene_graph}."),
    ("system", "Available tools: {tools}. Use them wisely. Actively use retrieval tools to come up with plausible answer."),
    ("system", "Tool names: {tool_names}"),  # for React agents
    ("user", "{query}"),
    ("assistant", "{agent_scratchpad}")  # for React agents
    ])




#-------------------------
# AGENT 2a, AGENT 2b,
"""

"""
#-------------------------
AGENT2a_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an agent that answers queries using tools. If you have gathered enough information, respond with:
    
    Thought: I now have enough information to answer.
    Final Answer: [Your answer]
    
    Otherwise, use this format:
    Thought: [Your reasoning]
    Action: [Tool name]
    Action Input: {{"query": {query}, "target_activity":{target_activity}, "target_scene_graph": {target_scene_graph} }}
    """),        

    ("system", "The user wants to perform a target activity. The user is in a space described by this scene graph. Both target activity and scene graph is given in user message. Only use entities in this scene graph."),
    ("system", "Available tools: {tools}. Use them when necessary."),
    ("system", "Tool names: {tool_names}"), 
    ("system",  "user target activity: {target_activity}"),
    ("system",  "user target scene graph: {target_scene_graph}"),
    ("user", "user query: {query}"), 
    ("assistant", "{agent_scratchpad}") 
])




AGENT2b_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an agent that answers queries using tools. If you have gathered enough information, respond with:
    
    Thought: I now have enough information to answer.
    Final Answer: [Your answer]
    
    Otherwise, use this format:
    Thought: [Your reasoning]
    Action: [Tool name]
    Action Input: {{"query": {query}, "target_activity":{target_activity}, "target_scene_graph": {target_scene_graph} }}
    """),        

    ("system", "The user wants to perform a target activity. The user is in a space described by this scene graph. Both target activity and scene graph is given in user message. Only use entities in this scene graph."),
    ("system", "Available tools: {tools}. Use them when necessary."),
    ("system", "Tool names: {tool_names}"), 
    ("system",  "user target activity: {target_activity}"),
    ("system",  "user target scene graph: {target_scene_graph}"),
    ("user", "user query: {query}"), 
    ("assistant", "{agent_scratchpad}") 
])







#-------------------------
# AGENT 3
"""

"""
#-------------------------
AGENT3_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an agent that answers queries using tools. If you have gathered enough information, respond with:
    
    Thought: I now have enough information to answer.
    Final Answer: [Your answer]
    
    Otherwise, use this format:
    Thought: [Your reasoning]
    Action: [Tool name]
    Action Input: {{"query": {query}, "target_activity":{target_activity}, "target_scene_graph": {target_scene_graph} }}
    """),        

    ("system", "The user wants to perform a target activity. The user is in a space described by this scene graph. Both target activity and scene graph is given in user message. Only use entities in this scene graph."),
    ("system", "Available tools: {tools}. Use them when necessary."),
    ("system", "Tool names: {tool_names}"), 
    ("system",  "user target activity: {target_activity}"),
    ("system",  "user target scene graph: {target_scene_graph}"),
    ("user", "user query: {query}"), 
    ("assistant", "{agent_scratchpad}") 
]
)
























#-------------------------
# AGENT 3
#-------------------------
# action_sequence_generation
prompt = f"You are an action sequence generator that breaks down activity in the query into step by step action. Please ONLY use entities of various states given by the query's spatial_information.\n You can also retrieve goalstep information in other environments using goalstep_retriever to see which activity might lead to which specific actions.\n You can also retrieve spatial information for other environments to see how entities and their state changes in other environment takes place.\n"

# action_sequence_validation
prompt = f"You are an action sequence validator. You have to check three things. First, you need to see whether the current action sequence fulfills the target activity given in the query. Second, you need to see whether the actions are performable with entities of various states given in the query as spatial_information. Third, you need to check if the sequence of actions are performed in logical step. When any of the three items in the checklist fails, you have to generate reasons why validation failed to the action_sequence_generation tool for regenerating answer.\n"








# define template
template0 = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 

    #Role:
    {role}
    #Question: 
    {question} 
    #Action_Sequence: 
    {action_sequence} 
    #Spatial_Layout
    {spatial_layout}
    """


template1 = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 

    #Role:
    {role}
    #Question: 
    {question} 
    #Action_Sequence: 
    {action_sequence} 
    #Spatial_Layout
    {spatial_layout}
    #Similar_Steps
    {relevant_actions}
    #Similar_Space_Example
    {relevant_space}
    """

template_baseline0 = """You are an assitant for question-answering tasks. Use the following pieces of retrieved context to answer the question. You must follow the rules provided

    #Role:
    {role}
    #Question:
    {question}
    #Goal:
    {goal}
    #Spatial_Layout:
    {spatial_layout}
    #Rules:
    {rules}
"""

template_baseline1 = """You are an assitant for question-answering tasks. Use the following pieces of retrieved context to answer the question. You must follow the rules provided. When answering questions, you can always look at the relevant_examples for valid question-answer pairs.

    #Role:
    {role}
    #Question:
    {question}
    #Goal:
    {goal}
    #Spatial_Layout:
    {spatial_layout}
    #Rules:
    {rules}
    #Relevant_Examples:
    {relevant_examples}
    """


#MESSAGES from other examples(AntGPT)====================
message = [
    {
        'role':'system',
        'content':'',
    }
]


message = [{'role':'system', 'content': f"Suppose a person has performed the given actions in the form of a sequence of action pairs, there should be no more than two words in each action pair Each action pair is defined by a {{verb}} and a {{noun}}, separated by a space. What will be the possible next 20 actions? \\You should follow the following rules: 1.For each action pairs, you can only choose the {{verb}} from the following words: [adjust, apply, arrange, attach, blow, break, carry, catch, clap, clean, climb, close, consume, count, cover, crochet, cut, detach, dig, dip, divide, draw, drill, drive, enter, feed, file, fill, fold, fry, give, grate, grind, hang, hit, hold, insert, inspect, iron, kick, knead, knit, lift, lock, loosen, mark, measure, mix, mold, move, open, operate, pack, paint, park, peel, pet, plant, play, point, pour, press, pull, pump, push, put, read, remove, repair, roll, sand, scoop, scrape, screw, scroll, search, serve, sew, shake, sharpen, shuffle, sieve, sit, smooth, spray, sprinkle, squeeze, stand, step, stick, stretch, swing, take, talk, throw, tie, tighten, tilt, touch, turn, turn, turn, uncover, unfold, unroll, unscrew, untie, walk, wash, water, wear, weld, wipe, write, zip] \\ 2. For each action pairs, you can only choose the {{noun}} from the following words: [apple, apron, arm, artwork, asparagus, avocado, awl, axe, baby, bacon, bag, baking, ball, ball, balloon, banana, bar, baseboard, basket, bat, bat, bathtub, batter, battery, bead, beaker, bean, bed, belt, bench, berry, beverage, bicycle, blanket, blender, block, blower, bolt, book, bookcase, bottle, bowl, bracelet, brake, brake, branch, bread, brick, broccoli, broom, brush, bubble, bucket, buckle, burger, butter, butterfly, button, cabbage, cabinet, calculator, caliper, camera, can, candle, canvas, car, card, cardboard, carpet, carrot, cart, cat, ceiling, celery, cello, cement, cereal, chaff, chain, chair, chalk, cheese, chicken, chip, chip, chip, chisel, chocolate, chopping, chopstick, cigarette, circuit, clamp, clay, clip, clock, cloth, coaster, coconut, coffee, coffee, colander, comb, computer, container, cooker, cookie, cork, corn, corner, countertop, crab, cracker, crayon, cream, crochet, crowbar, cucumber, cup, curtain, cushion, cutter, decoration, derailleur, detergent, dice, dishwasher, dog, door, doorbell, dough, dough, doughnut, drawer, dress, drill, drill, drum, dumbbell, dust, duster, dustpan, egg, eggplant, engine, envelope, eraser, facemask, fan, faucet, fence, file, filler, filter, fish, fishing, flash, floor, flour, flower, foam, foil, food, foot, fork, fridge, fries, fuel, funnel, game, garbage, garlic, gasket, gate, gauge, gauze, gear, generator, ginger, glass, glasses, glove, glue, glue, golf, gourd, grain, grape, grapefruit, grass, grater, grill, grinder, guava, guitar, hair, hammer, hand, handle, hanger, hat, hay, haystack, head, headphones, heater, helmet, hinge, hole, horse, hose, house, ice, ice, ink, iron, jack, jacket, jug, kale, ketchup, kettle, key, keyboard, knife, label, ladder, leaf, leash, leg, lemon, lever, lid, light, lighter, lime, lock, lubricant, magnet, mango, manure, mask, mat, matchstick, meat, medicine, metal, microscope, microwave, milk, mirror, mixer, mold, money, mop, motorcycle, mouse, mouth, mower, multimeter, mushroom, nail, nail, nail, napkin, necklace, needle, net, nozzle, nut, nut, oil, okra, onion, oven, paddle, paint, paint, paintbrush, palette, pan, pancake, panel, pants, papaya, paper, pasta, paste, pastry, pea, peanut, pear, pedal, peel, peeler, peg, pen, pencil, pepper, phone, photo, piano, pickle, picture, pie, pillow, pilot, pin, pipe, pizza, planer, plant, plate, playing, plier, plug, pole, popcorn, pot, pot, potato, pump, pumpkin, purse, puzzle, rack, radio, rail, rake, razor, remote, rice, ring, rod, rolling, root, rope, router, rubber, ruler, sand, sander, sandpaper, sandwich, sauce, sausage, saw, scarf, scissors, scoop, scraper, screw, screwdriver, sculpture, seasoning, seed, set, sewing, sharpener, shears, sheet, shelf, shell, shirt, shoe, shovel, shower, sickle, sieve, sink, sketch, skirt, slab, snorkel, soap, sock, socket, sofa, soil, solder, soup, spacer, spatula, speaker, sphygmomanometer, spice, spinach, spirit, sponge, spoon, spray, spring, squeezer, stairs, stamp, stapler, steamer, steering, stick, sticker, stock, stone, stool, stove, strap, straw, string, stroller, switch, syringe, table, tablet, taco, tape, tape, tea, teapot, television, tent, test, tie, tile, timer, toaster, toilet, toilet, tomato, tongs, toolbox, toothbrush, toothpick, torch, towel, toy, tractor, trash, tray, treadmill, tree, trimmer, trowel, truck, tweezer, umbrella, undergarment, vacuum, vacuum, valve, vase, video, violin, wall, wallet, wallpaper, washing, watch, water, watermelon, weighing, welding, wheat, wheel, wheelbarrow, whisk, window, windshield, wiper, wire, wood, worm, wrapper, wrench, yam, yeast, yoghurt, zipper, zucchini].\\Remember the output must be exact 20 actions in the form of {{verb}} and a {{noun}}, which means there are 19 ',' in the output.\n"},]



demos = [
    "tie leaf, carry leaf, put plant, adjust cloth, take plant, put plant, take sickle, take sickle ##Q1: What's the scene according to previous actions? Q2: What are the future 20 actions based on the scene from Q1 and previous actions? => The scene is gardening. Future 20 actions are: cut plant, put sickle, take leaf, stretch rubber, take sickle, cut plant, hold plant, put sickle, take rubber, pull rubber, take rubber, tie rubber, move plant, take rubber, pull rubber, put plant, hold plant, cut plant, cut plant, hold plant ###\n",

    "put cement, wipe mold, arrange mold, turn mold, put mold, take soil, pour mold, remove mold ##Q1: What's the scene according to previous actions? Q2: What are the future 20 actions based on the scene from Q1 and previous actions? => The scene is making bricks. Future 20 actions are: put mold, wipe floor, cut cement, mix cement, arrange mold, put cement, remove cement, put cement, wipe cement, carry mold, put mold, turn mold, put mold, pour sand, put mold, cut clay, arrange mold, put clay, remove clay, carry mold ###\n",
]






