from astar.search import AStar

class AStarPlanner:
    def __init__(self, free_positions, step_size) -> None:
        self.create_map(free_positions, step_size)

    # Create a Map Representation
    def create_map(self, free_positions, step_size):
        self.step_size = step_size        
        self.min_x = 0
        self.max_x = 0
        self.min_z = 0
        self.max_z = 0
        first_point = True
        for point in free_positions:
            if first_point:
                self.min_x = point['x']
                self.max_x = point['x']
                self.min_z = point['z']
                self.max_z = point['z']
                first_point = False
            else:
                if self.min_x > point['x']:
                    self.min_x = point['x']
                elif self.max_x < point['x']:
                    self.max_x = point['x']
                if self.min_z > point['z']:
                    self.min_z = point['z']
                elif self.max_z < point['z']:
                    self.max_z = point['z']

        self.num_rows = int((self.max_x - self.min_x) / step_size) + 1
        self.num_cols = int((self.max_z - self.min_z) / step_size) + 1
        #print("num rows = ", num_rows)
        #print("num cols = ", num_cols)
        self.env_map = [[1 for c in range(self.num_cols)] for r in range(self.num_rows)]
                
        for point in free_positions:
            r, c = self.get_row_col(point['x'], point['z'])
            self.env_map[r][c] = 0

    ## Run A-Star Planner 
    
    def get_path(self, start, goal):
        start_rc = self.get_row_col(start[0], start[1])
        goal_rc = self.get_row_col(goal[0], goal[1])

        path_rc = AStar(self.env_map).search(start_rc, goal_rc)

        return [self.get_x_z(p[0], p[1]) for p in path_rc]

    ## Helper Functions

    def get_row_col(self, x, z):
        r = (x - self.min_x) / self.step_size
        c = (z - self.min_z) / self.step_size
        return int(r), int(c)

    def get_x_z(self, r, c):
        x = self.min_x + r*self.step_size
        z = self.min_z + c*self.step_size
        return x, z
    
    


