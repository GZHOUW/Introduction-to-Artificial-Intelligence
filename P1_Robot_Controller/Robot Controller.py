import helper
env_data = helper.fetch_maze()

# Number of rows
rows = len(env_data) 

# Number of columns
columns = len(env_data[0]) 

# Extract the element of the third row and sixth column
row_3_col_6 = env_data[2][5]

print("The maze has", rows, "rows", columns, "columns，the element of the third row and sixth column is", row_3_col_6)

# Calculate the number of barriers in the first row
number_of_barriers_row1 = env_data[0].count(2)

# Calculate the number of barriers in the third row
col3 = []
for element in env_data:
    col3.append(element[2])
number_of_barriers_col3 = col3.count(2)

print("The first row has", number_of_barriers_row1, "barriers，the third colomn has", number_of_barriers_col3, "barriers.")

for r_element in env_data:
    for c_element in r_element:
        if c_element == 1:
            tup1 = (env_data.index(r_element), r_element.index(c_element))
        elif c_element == 3:
            tup2 = (env_data.index(r_element), r_element.index(c_element))
    
loc_map = {'start':tup1, 'destination':tup2}

robot_current_loc = loc_map['start'] #Save the current location of the robot

def is_move_valid_sepcial(loc, act):
    if act == 'd':
        if loc[0] == len(env_data)-1: 
            return False
        elif env_data[loc[0]+1][loc[1]] == 2:
            return False
        else:
            return True
    elif act == 'u':
        if loc[0] == 0:
            return False
        elif  env_data[loc[0]-1][loc[1]] == 2:
            return False
        else:
            return True
    elif act == 'l':
        if loc[1] == 0:
            return False
        elif env_data[loc[0]][loc[1]-1] == 2:
            return False
        else:
            return True
    elif act == 'r':
        if loc[1] == len(env_data[0])-1:
            return False
        elif  env_data[loc[0]][loc[1]+1] == 2:
            return False
        else:
            return True

def is_move_valid(env_data, loc, act):
    if act == 'd':
        if loc[0] == len(env_data)-1:
            return False
        elif env_data[loc[0]+1][loc[1]] == 2:
            return False
        else:
            return True
    elif act == 'u':
        if loc[0] == 0:
            return False
        elif  env_data[loc[0]-1][loc[1]] == 2:
            return False
        else:
            return True
    elif act == 'l':
        if loc[1] == 0:
            return False
        elif env_data[loc[0]][loc[1]-1] == 2:
            return False
        else:
            return True
    elif act == 'r':
        if loc[1] == len(env_data[0])-1:
            return False
        elif  env_data[loc[0]][loc[1]+1] == 2:
            return False
        else:
            return True

def valid_actions(env_data,loc):
    act = ['u','d','l','r']
    avaliable_act = []
    for direction in act:
        if is_move_valid(env_data, loc, direction) == True:
            avaliable_act.append(direction)
    return avaliable_act

 def move_robot(loc,act):
    if act == 'u':
        new_loc = (loc[0]-1, loc[1])
    elif act == 'd':
        new_loc = (loc[0]+1, loc[1])
    elif act == 'l':
        new_loc = (loc[0], loc[1]-1)
    elif act == 'r':
        new_loc = (loc[0], loc[1]+1)
    return new_loc

import random
def random_choose_actions(env_data,loc):
    for i in range(300):
        choice = random.choice(valid_actions(env_data,loc))
        new_loc = move_robot(loc,choice)
        if env_data[new_loc[0]][new_loc[1]] == 3:
            print("Found the treasure! Found in"+ str(i) + "rounds！")
            break
        loc = new_loc

# Run
random_choose_actions(env_data, robot_current_loc)
