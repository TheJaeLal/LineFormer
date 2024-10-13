def bresenham_hv(x1, y1, x2, y2):
    # Calculate deltas
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    # Determine the direction of the line
    if dx > dy:  # Horizontal dominant line
        # Initialize error
        error = 0
        y = y1
        step_y = 1 if y2 > y1 else -1
        
        # Iterate over x
        for x in range(x1, x2 + 1 if x2 > x1 else x2 - 1, 1 if x2 > x1 else -1):
            plot(x, y)  # Plot the current pixel
            
            # Accumulate error
            error += dy
            
            # Move vertically if error surpasses threshold
            if error >= dx:
                y += step_y
                error -= dx
    
    else:  # Vertical dominant line
        # Initialize error
        error = 0
        x = x1
        step_x = 1 if x2 > x1 else -1
        
        # Iterate over y
        for y in range(y1, y2 + 1 if y2 > y1 else y2 - 1, 1 if y2 > y1 else -1):
            plot(x, y)  # Plot the current pixel
            
            # Accumulate error
            error += dx
            
            # Move horizontally if error surpasses threshold
            if error >= dy:
                x += step_x
                error -= dy
