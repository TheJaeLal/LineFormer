import pandas as pd

def get_kaplan_meier_data_from_events(events:list, group:int):
    times = [event[0] for event in events]
    number_at_risk = [event[1] for event in events]
    censoring_status = [group] * len(events)  # Assuming all events are observed (censoring status = 1)

    # Create a DataFrame
    df = pd.DataFrame({
        'Time': times,
        'Censoring_Status': censoring_status,
        'Number_at_Risk': number_at_risk
    })

    return df


def map_px_to_plot_coordinates(intersection_points, plot_start_coord, plot_end_coord):

    x_pixel_min, y_pixel_max = intersection_points[0]
    x_pixel_max, y_pixel_min = intersection_points[-1]

    x_coord_min, y_coord_max = plot_start_coord
    x_coord_max, y_coord_min = plot_end_coord

    # Rescaling function
    def rescale(value, old_min, old_max, new_min, new_max):
        return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    # Empty list to store the results
    mapped_coordinates = []

    # Loop over the pixel values
    for x, y in intersection_points:
        new_x = rescale(x, x_pixel_min, x_pixel_max, x_coord_min, x_coord_max)
        new_y = rescale(y, y_pixel_min, y_pixel_max, y_coord_min, y_coord_max)
        mapped_coordinates.append((new_x, new_y))

    return mapped_coordinates

