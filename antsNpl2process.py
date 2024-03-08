"""
Module managing an ant colony in a labyrinth.
"""
import numpy as np
import maze
import pheromone
import direction as d
import pygame as pg

UNLOADED, LOADED = False, True

exploration_coefs = 0.


class Colony:
    """
    Represent an ant colony. Ants are not individualized for performance reasons!

    Inputs :
        nb_ants  : Number of ants in the anthill
        pos_init : Initial positions of ants (anthill position)
        max_life : Maximum life that ants can reach
    """
    def __init__(self, nb_ants, pos_init, max_life):
        # Each ant has is own unique random seed
        self.seeds = np.arange(1, nb_ants+1, dtype=np.int64)
        # State of each ant : loaded or unloaded
        self.is_loaded = np.zeros(nb_ants, dtype=np.int8)
        # Compute the maximal life amount for each ant :
        #   Updating the random seed :
        self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
        # Amount of life for each ant = 75% à 100% of maximal ants life
        self.max_life = max_life * np.ones(nb_ants, dtype=np.int32)
        self.max_life -= np.int32(max_life*(self.seeds/2147483647.))//4
        # Ages of ants : zero at beginning
        self.age = np.zeros(nb_ants, dtype=np.int64)
        # History of the path taken by each ant. The position at the ant's age represents its current position.
        self.historic_path = np.zeros((nb_ants, max_life+1, 2), dtype=np.int16)
        self.historic_path[:, 0, 0] = pos_init[0]
        self.historic_path[:, 0, 1] = pos_init[1]
        # Direction in which the ant is currently facing (depends on the direction it came from).
        self.directions = d.DIR_NONE*np.ones(nb_ants, dtype=np.int8)

    def load_sprites(self):
        self.sprites = []
        img = pg.image.load("ants.png").convert_alpha()
        for i in range(0, 32, 8):
            self.sprites.append(pg.Surface.subsurface(img, i, 0, 8, 8))

    def return_to_nest(self, loaded_ants, pos_nest, food_counter):
        """
        Function that unloads ants that have food and are located at the nest

        Inputs :
            loaded_ants: Indices of ants carrying food
            pos_nest: Position of the nest where ants should go
            food_counter: Current quantity of food in the nest

        Returns the new quantity of food
        """
        self.age[loaded_ants] -= 1 # Loaded ants dont age

        in_nest_tmp = self.historic_path[loaded_ants, self.age[loaded_ants], :] == pos_nest
        if in_nest_tmp.any():
            in_nest_loc = np.nonzero(np.logical_and(in_nest_tmp[:, 0], in_nest_tmp[:, 1]))[0]
            if in_nest_loc.shape[0] > 0:
                in_nest = loaded_ants[in_nest_loc]
                self.is_loaded[in_nest] = UNLOADED
                self.age[in_nest] = 0 # Age reset
                food_counter += in_nest_loc.shape[0] # Food added
        return food_counter

    def explore(self, unloaded_ants, the_maze, pos_food, pos_nest, pheromones):
        """
        Management of unloaded ants exploring the maze.

        Inputs:
            unloadedAnts: Indices of ants that are not loaded
            maze        : The maze in which ants move
            posFood     : Position of food in the maze
            posNest     : Position of the ants' nest in the maze
            pheromones  : The pheromone map (which also has ghost cells for
                          easier edge management)

        Outputs: None
        """
        # Update of the random seed (for manual pseudo-random) applied to all unloaded ants
        self.seeds[unloaded_ants] = np.mod(16807*self.seeds[unloaded_ants], 2147483647)

        # Calculating possible exits for each ant in the maze:
        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.WEST) > 0

        # Reading neighboring pheromones:
        north_pos = np.copy(old_pos_ants)
        north_pos[:, 1] += 1
        north_pheromone = pheromones.pheromon[north_pos[:, 0], north_pos[:, 1]]*has_north_exit

        east_pos = np.copy(old_pos_ants)
        east_pos[:, 0] += 1
        east_pos[:, 1] += 2
        east_pheromone = pheromones.pheromon[east_pos[:, 0], east_pos[:, 1]]*has_east_exit

        south_pos = np.copy(old_pos_ants)
        south_pos[:, 0] += 2
        south_pos[:, 1] += 1
        south_pheromone = pheromones.pheromon[south_pos[:, 0], south_pos[:, 1]]*has_south_exit

        west_pos = np.copy(old_pos_ants)
        west_pos[:, 0] += 1
        west_pheromone = pheromones.pheromon[west_pos[:, 0], west_pos[:, 1]]*has_west_exit

        max_pheromones = np.maximum(north_pheromone, east_pheromone)
        max_pheromones = np.maximum(max_pheromones, south_pheromone)
        max_pheromones = np.maximum(max_pheromones, west_pheromone)

        # Calculating choices for all ants not carrying food (for others, we calculate but it doesn't matter)
        choices = self.seeds[:] / 2147483647.

        # Ants explore the maze by choice or if no pheromone can guide them:
        ind_exploring_ants = np.nonzero(
            np.logical_or(choices[unloaded_ants] <= exploration_coefs, max_pheromones[unloaded_ants] == 0.))[0]
        if ind_exploring_ants.shape[0] > 0:
            ind_exploring_ants = unloaded_ants[ind_exploring_ants]
            valid_moves = np.zeros(choices.shape[0], np.int8)
            nb_exits = has_north_exit * np.ones(has_north_exit.shape) + has_east_exit * np.ones(has_east_exit.shape) + \
                has_south_exit * np.ones(has_south_exit.shape) + has_west_exit * np.ones(has_west_exit.shape)
            while np.any(valid_moves[ind_exploring_ants] == 0):
                # Calculating indices of ants whose last move was not valid:
                ind_ants_to_move = ind_exploring_ants[valid_moves[ind_exploring_ants] == 0]
                self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
                # Choosing a random direction:
                dir = np.mod(self.seeds[ind_ants_to_move], 4)
                old_pos = self.historic_path[ind_ants_to_move, self.age[ind_ants_to_move], :]
                new_pos = np.copy(old_pos)
                new_pos[:, 1] -= np.logical_and(dir == d.DIR_WEST,
                                                has_west_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 1] += np.logical_and(dir == d.DIR_EAST,
                                                has_east_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 0] -= np.logical_and(dir == d.DIR_NORTH,
                                                has_north_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 0] += np.logical_and(dir == d.DIR_SOUTH,
                                                has_south_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                # Valid move if we didn't stay in place due to a wall
                valid_moves[ind_ants_to_move] = np.logical_or(new_pos[:, 0] != old_pos[:, 0], new_pos[:, 1] != old_pos[:, 1])
                # and if we're not in the opposite direction of the previous move (and if there are other exits)
                valid_moves[ind_ants_to_move] = np.logical_and(
                    valid_moves[ind_ants_to_move],
                    np.logical_or(dir != 3-self.directions[ind_ants_to_move], nb_exits[ind_ants_to_move] == 1))
                # Calculating indices of ants whose move we just validated:
                ind_valid_moves = ind_ants_to_move[np.nonzero(valid_moves[ind_ants_to_move])[0]]
                # For these ants, we update their positions and directions
                self.historic_path[ind_valid_moves, self.age[ind_valid_moves] + 1, :] = new_pos[valid_moves[ind_ants_to_move] == 1, :]
                self.directions[ind_valid_moves] = dir[valid_moves[ind_ants_to_move] == 1]

        ind_following_ants = np.nonzero(np.logical_and(choices[unloaded_ants] > exploration_coefs,
                                                       max_pheromones[unloaded_ants] > 0.))[0]
        if ind_following_ants.shape[0] > 0:
            ind_following_ants = unloaded_ants[ind_following_ants]
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, :] = \
                self.historic_path[ind_following_ants, self.age[ind_following_ants], :]
            max_east = (east_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 1] += \
                max_east * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_west = (west_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 1] -= \
                max_west * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_north = (north_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 0] -= max_north * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_south = (south_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 0] += max_south * np.ones(ind_following_ants.shape[0], dtype=np.int16)

        # Aging one unit for the age of ants not carrying food
        if unloaded_ants.shape[0] > 0:
            self.age[unloaded_ants] += 1

        # Killing ants at the end of their life:
        ind_dying_ants = np.nonzero(self.age == self.max_life)[0]
        if ind_dying_ants.shape[0] > 0:
            self.age[ind_dying_ants] = 0
            self.historic_path[ind_dying_ants, 0, 0] = pos_nest[0]
            self.historic_path[ind_dying_ants, 0, 1] = pos_nest[1]
            self.directions[ind_dying_ants] = d.DIR_NONE

        # For ants reaching food, we update their states:
        ants_at_food_loc = np.nonzero(np.logical_and(self.historic_path[unloaded_ants, self.age[unloaded_ants], 0] == pos_food[0],
                                                     self.historic_path[unloaded_ants, self.age[unloaded_ants], 1] == pos_food[1]))[0]
        if ants_at_food_loc.shape[0] > 0:
            ants_at_food = unloaded_ants[ants_at_food_loc]
            self.is_loaded[ants_at_food] = True

    def advance(self, the_maze, pos_food, pos_nest, pheromones, food_counter=0):
        loaded_ants = np.nonzero(self.is_loaded == True)[0]
        unloaded_ants = np.nonzero(self.is_loaded == False)[0]
        if loaded_ants.shape[0] > 0:
            food_counter = self.return_to_nest(loaded_ants, pos_nest, food_counter)
        if unloaded_ants.shape[0] > 0:
            self.explore(unloaded_ants, the_maze, pos_food, pos_nest, pheromones)
            return food_counter


    def display(self, screen):
        [screen.blit(self.sprites[self.directions[i]], (8*self.historic_path[i, self.age[i], 1], 8*self.historic_path[i, self.age[i], 0])) for i in range(self.directions.shape[0])]
    
    def updatePheromones(self, pheromones, the_maze):
        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.WEST) > 0
        # Marking pheromones:
        [pheromones.mark(self.historic_path[i, self.age[i], :],
                         [has_north_exit[i], has_east_exit[i], has_west_exit[i], has_south_exit[i]]) for i in range(self.directions.shape[0])]

if __name__ == "__main__":
    import sys
    import time
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    # Groupe des communicateurs pour le calcul des fourmis parallélisé
    group_ant_calc = comm.Get_group().Incl([i for i in range(1,size)])
    comm_group = comm.Create(group_ant_calc)
    if rank in range(1,size):
        ant_init_data = np.empty(7, dtype=np.int32)



# - - - - - - - - - - - - - - - - - - - - -
    
# - - - - - Comm qui affiche  - - - - - - - 

# - - - - - - - - - - - - - - - - - - - - -

# Possible de poser alpha = 1 pour rendre plus facile
    if rank == 0:
        pg.init()

        # Reception des données
        intBuffer = np.empty(8, dtype=np.int32)
        floatBuffer = np.empty(2, dtype=np.float32)
        comm.Recv(intBuffer, source=1)
        comm.Recv(floatBuffer, source=1)
        size_laby = np.empty(2, dtype=np.int32)
        pos_nest = np.empty(2, dtype=np.int32)
        pos_food = np.empty(2, dtype=np.int32)
        size_laby[0] = intBuffer[0]
        size_laby[1] = intBuffer[1]
        nb_ants = intBuffer[2]
        max_life = intBuffer[3]
        pos_nest[0] = intBuffer[4]
        pos_nest[1] = intBuffer[5]
        pos_food[0] = intBuffer[6]
        pos_food[1] = intBuffer[7]

        alpha = floatBuffer[0]
        beta = floatBuffer[1]

        resolution = size_laby[1]*8, size_laby[0]*8
        screen = pg.display.set_mode(resolution)

        # Initialisation des objets du front
        a_maze = maze.Maze(size_laby, 12345)
        a_maze.load_patterns()
        ants = Colony(nb_ants, pos_nest, max_life)
        ants.load_sprites()
        pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
        mazeImg = a_maze.display()

        while True:
            # Réception continue des données
            comm.Recv(pherom.pheromon, source=1)
            comm.Recv(ants.historic_path, source=1)
            comm.Recv(ants.age, source=1)
            comm.Recv(ants.directions, source=1)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit(0)
            pherom.display(screen)
            screen.blit(mazeImg, (0, 0))
            ants.display(screen)
            pg.display.update()
            

# - - - - - - - - - - - - - - - - - - - - -

# - - - - - Comms pour le calcul  - - - - -

# - - - - - - - - - - - - - - - - - - - - -
    if rank != 0:
        
        # Initialisation des buffers pour Bcast
        ant_init_data = np.empty(9, dtype=np.int32)
        float_buffer = np.empty(2, dtype=np.float32)
        total_historic_buffer = None
        total_food_buffer = None
        local_food_buffer = np.empty(1, dtype=np.int32)

        # Initialisation Paramètres
        if rank == 1:
    
            size_laby = 25, 25
            nb_ants = size_laby[0]*size_laby[1]//4
            nb_ants -= nb_ants%(size-1)
            max_life = 500
            pos_food = size_laby[0]-1, size_laby[1]-1
            pos_nest = 0, 0
            maze_seed = 12345
            alpha = 0.9
            beta  = 0.99
            a_maze = maze.Maze(size_laby, maze_seed)
            total_food_counter = 0


            # Transmission initiale des données au front
            intBuffer = np.array([size_laby[0], size_laby[1], nb_ants, max_life, pos_nest[0], pos_nest[1],
                                    pos_food[0], pos_food[1]], dtype=np.int32)
            floatBuffer = np.array([alpha, beta], dtype = np.float32)
            total_food_buffer = np.empty((size - 1,1), dtype=np.int32)
            comm.Send(intBuffer, 0)
            comm.Send(floatBuffer, 0)


            # Données à transmettre pour les communicateurs "fourmis"
            ant_init_data = np.array([size_laby[0], size_laby[1], nb_ants, max_life, pos_food[0], pos_food[1], pos_nest[0], pos_nest[1], maze_seed],
                                dtype=np.int32)
            float_buffer = np.array([alpha, beta], dtype=np.float32)
            total_ants = Colony(nb_ants, pos_nest, max_life)
            unloaded_ants = np.array(range(nb_ants))



        #Récupération des paramètres 
        comm_group.Bcast(ant_init_data, root=0)
        comm_group.Bcast(float_buffer, root=0)
        if rank != 1:
            size_laby = ant_init_data[0], ant_init_data[1]
            nb_ants = ant_init_data[2]
            max_life = ant_init_data[3]
            pos_food = ant_init_data[4], ant_init_data[5]
            pos_nest = ant_init_data[6], ant_init_data[7]
            maze_seed = ant_init_data[8]
            alpha = float_buffer[0]
            beta = float_buffer[1]
        a_maze = maze.Maze(size_laby, maze_seed)
        pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
        nb_local_ants = nb_ants//(size-1)
        local_ants = Colony(nb_local_ants, pos_nest, max_life)
        local_ants.seeds += rank # Restauration de l'aléatoire
        total_historic_buffer = None
        total_age_buffer = None
        total_directions_buffer = None
        total_isLoaded_buffer = None
        if rank == 1:
            total_historic_buffer =  total_ants.historic_path.reshape([size - 1, nb_local_ants, max_life+1, 2])
            total_age_buffer = np.empty((size-1, nb_ants//(size-1)), dtype=np.int64 )
            total_directions_buffer = np.empty((size-1, nb_ants//(size-1)), dtype=np.int8 )
            total_isLoaded_buffer = np.empty((size-1, nb_ants//(size-1)), dtype=np.int8 )
        
        comm_group.Scatter(total_historic_buffer, local_ants.historic_path, root=0)
        
        # Boucle principale des Backends
        snapshop_taken = False
        img_counter = 0
        t_start = time.time()
        while img_counter < 5000:
            # Calcul Parallèle
            local_food_counter = 0
            comm_group.Bcast(pherom.pheromon, root=0)
            local_food_counter = local_ants.advance(a_maze, pos_food, pos_nest, pherom, local_food_counter)
            if local_food_counter == None:
                local_food_counter = 0

            # Envoi des données au Back Maître
            local_food_buffer[0] = local_food_counter
            comm_group.Gather(local_food_buffer, total_food_buffer, root=0)
            comm_group.Gather(local_ants.historic_path, total_historic_buffer, root=0)
            comm_group.Gather(local_ants.age, total_age_buffer, root=0)
            comm_group.Gather(local_ants.directions, total_directions_buffer, root=0)
            comm_group.Gather(local_ants.is_loaded, total_isLoaded_buffer, root=0)


            if rank == 1:
                deb = time.time()
                total_ants.historic_path = total_historic_buffer.reshape(nb_ants, max_life+1, 2)
                total_ants.age = total_age_buffer.reshape(nb_ants)
                total_ants.directions = total_directions_buffer.reshape(nb_ants)
                total_ants.is_loaded = total_isLoaded_buffer.reshape(nb_ants)
                total_ants.updatePheromones(pherom, a_maze)
                pherom.do_evaporation(pos_food)
                total_food_counter += total_food_buffer.sum()
                total_ants.historic_path = total_historic_buffer.reshape((nb_ants,max_life+1, 2))

                # Envoi de données au Front
                comm.Send(pherom.pheromon, dest=0)
                comm.Send(total_ants.historic_path, dest=0)
                comm.Send(total_ants.age, dest=0)
                comm.Send(total_ants.directions, dest=0)

                if total_food_counter == 1 and not snapshop_taken:
                    # pg.image.save(screen, "MyFirstFood.png")
                    snapshop_taken = True
                # pg.time.wait(500)
                end = time.time()
                print(f"FPS : {1./(end-deb):6.2f}, nourriture : {total_food_counter:7d}", end='\r')
            img_counter += 1
        t_end = time.time()
        print("tps_ref : ", t_end-t_start)


# - - - - - - - - - - - - - - - - - - - - -
    
# - - - - Libération des ressources - - - - 

# - - - - - - - - - - - - - - - - - - - - -
if group_ant_calc:
    group_ant_calc.Free()