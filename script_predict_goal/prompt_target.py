import json
import argparse

"""
used only for storing examples
"""
message = [
    {
        'role':'system',
        'content':'',
    }
]

demos = [
    "tie leaf, carry leaf, put plant, adjust cloth, take plant, put plant, take sickle, take sickle ##Q1: What's the scene according to previous actions? Q2: What are the future 20 actions based on the scene from Q1 and previous actions? => The scene is gardening. Future 20 actions are: cut plant, put sickle, take leaf, stretch rubber, take sickle, cut plant, hold plant, put sickle, take rubber, pull rubber, take rubber, tie rubber, move plant, take rubber, pull rubber, put plant, hold plant, cut plant, cut plant, hold plant ###\n",

    "put cement, wipe mold, arrange mold, turn mold, put mold, take soil, pour mold, remove mold ##Q1: What's the scene according to previous actions? Q2: What are the future 20 actions based on the scene from Q1 and previous actions? => The scene is making bricks. Future 20 actions are: put mold, wipe floor, cut cement, mix cement, arrange mold, put cement, remove cement, put cement, wipe cement, carry mold, put mold, turn mold, put mold, pour sand, put mold, cut clay, arrange mold, put clay, remove clay, carry mold ###\n",
]

if __name__== "__main__":
    print("trying to be better")