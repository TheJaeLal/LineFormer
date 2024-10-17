from collections import Counter



def vote_on_line(inline_group, axis):
    voted_line = Counter([item[axis] for item in inline_group]).most_common(1)[0][0]
    return voted_line


def get_inline_group_info(group_id, inline_group, voted_line, axis):
    start_pos = list(inline_group[0])
    start_pos[axis] = voted_line
    start_pos = tuple(start_pos)

    end_pos = list(inline_group[-1])
    end_pos[axis] = voted_line
    end_pos = tuple(end_pos)

    inline_group_info = {
        'group_id': group_id,
        'start_pos': start_pos,
        'end_pos': end_pos,
        'most_common_value': voted_line
    }

    return inline_group_info


def get_inline_groups(key_points, max_consecutive_increase_count=4, axis_to_group='y'):
    inline_groups_info = []
    current_inline_group = []
    consecutive_increase_count = 0

    prev_line = None

    axis = 1 if axis_to_group == 'x' else 0

    for kp in key_points:

        if len(current_inline_group) == 0:
            current_inline_group.append(kp)
            prev_line = kp[axis]
            continue
        
        current_inline_group.append(kp)

        if kp[axis] > prev_line:
            consecutive_increase_count += 1
        else:
            consecutive_increase_count = 0

        # Start new inline group
        if consecutive_increase_count >= max_consecutive_increase_count:
            info = get_inline_group_info(
                group_id=len(inline_groups_info), 
                inline_group=current_inline_group, 
                voted_line=prev_line, 
                axis=axis
            )
            inline_groups_info.append(info)
            current_inline_group = [kp]
            consecutive_increase_count = 0

        prev_line = vote_on_line(current_inline_group, axis)


    # Fill in last group
    if current_inline_group:
        info = get_inline_group_info(
            group_id=len(inline_groups_info), 
            inline_group=current_inline_group, 
            voted_line=prev_line, 
            axis=axis
        )
        inline_groups_info.append(info)


    return inline_groups_info





def process_groups(my_list, max_consecutive_increase_count=4, axis='y'):
    """
    Process groups of entries in `my_list` based on consecutive increases in the x or y values.

    Parameters:
        my_list (list of dicts): A list of dictionaries containing 'x' and 'y' values.
        max_consecutive_increase_count (int): The number of consecutive increases that trigger a new group.
        axis (str): Either 'x' or 'y' to specify which axis to use for processing ('x' or 'y').
        
    Returns:
        list: A list of processed groups, where values in the axis are adjusted based on most common value.
    """
    groups = []
    groups_info = []
    current_group = []
    consecutive_increase_count = 0
    prev_value = None  # This will hold the previous value of the axis (either x or y)

    # Choose the key based on the axis ('x' or 'y')
    axis_key = axis

    for i, entry in enumerate(my_list):
        if not current_group:
            # Start a new group
            current_group.append(entry)
            prev_value = entry[axis_key]  # Initialize the value as the first one
            continue

        # Check if the current value on the chosen axis is greater than the previous value
        if entry[axis_key] > prev_value:
            consecutive_increase_count += 1
        else:
            consecutive_increase_count = 0

        # Add the current entry to the current group
        current_group.append(entry)

        # If there are 'max_consecutive_increase_count' consecutive increases, close the current group
        if consecutive_increase_count >= max_consecutive_increase_count:
            # Get the most common value of the axis in the current group
            axis_values = [item[axis_key] for item in current_group]
            most_common_value = Counter(axis_values).most_common(1)[0][0]

            # Set all axis values in the current group to the most common value
            for entry in current_group:
                entry[axis_key] = most_common_value

            # Add the group to the list of groups
            groups.append(current_group)


            start_pos = (current_group[0]['x'], current_group[0]['y'])

            # Start a new group with the current item
            current_group = [entry]
            prev_value = entry[axis_key]  # Update the previous value for the new group
            consecutive_increase_count = 0

            groups_info.append({
                'group_id': len(groups),
                'start_pos': start_pos,
                'end_pos': (entry['x'], entry['y']),
                'most_common_value': prev_value
            })

        # Update the previous value to the most common value in the current group
        prev_value = Counter([item[axis_key] for item in current_group]).most_common(1)[0][0]

    # Process the last group
    if current_group:
        axis_values = [item[axis_key] for item in current_group]
        most_common_value = Counter(axis_values).most_common(1)[0][0]
        for entry in current_group:
            entry[axis_key] = most_common_value

        groups.append(current_group)

        start_pos = (current_group[0]['x'], current_group[0]['y'])
        groups_info.append({
            'group_id': len(groups),
            'start_pos': start_pos,
            'end_pos': (entry['x'], entry['y']),
            'most_common_value': prev_value
        })

    return my_list, groups_info

