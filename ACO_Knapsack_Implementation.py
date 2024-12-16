# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

def Main_Knap(Ant_Population, Evaporation_Variable, Deposition_Variable, Total_Generations, AntPop):
    """Initializes the knapsack function"""

    # Read weights from file
    Money_Sack, Peak_Capacity = File_Reader('/content/drive/MyDrive/BankProblem.txt')  # Read input data from a file.
    Sack_Items = Money_Sack.shape[0]  # Number of sacks (items).

    # Set up sacks
    Money_Sack = Money_Sack[np.argsort(Money_Sack[:, 0])]  # Sort sacks by weight in ascending order.

    # Initiate pheromone values
    Pheromone_Variable = np.ones(Sack_Items, dtype=np.float32)  # Initialize pheromone matrix with ones.

    # Create ant skeleton, Candidate vector and Fitness vector
    Ant_Number = Ant_Population  # Set the number of ants (population size).
    Ant_Fitness = np.zeros(Ant_Number, dtype=np.uint16)  # Initialize fitness vector.

    Peak_Value = [np.zeros(Total_Generations, dtype=np.float32), None]  # Initialize global max with zeros (Best solution {value, [path]})
    
    for Main_generation in range(Total_Generations):  # Main loop for generations
        # Let the ants make their paths
        Each_Ant_Path = []
        
        for ant in range(Ant_Number):  # from 1 to total of ants
            Main_Candidates = Pheromone_Variable.copy()  # Initialize candidates with pheromone levels.
            Path_Of_Ant = np.zeros(Sack_Items, dtype=np.uint8)  # Initialize path for the ant.
            Main_Path_Node = 0  # Start from the first node.
            Sum_Of_Weights = 0  # Initialize weight of the selected items to 0.
            Sum_Of_Candidates_ = np.sum(Main_Candidates)  # Sum of all pheromone levels (probability distribution).
            
            while Sum_Of_Candidates_ != 0:  # While there are valid candidates to choose from.
                Index_Of_Sack = Random_Index_Generator(Main_Candidates, Sum_Of_Candidates_)  # Get a random index based on pheromone levels.
                Path_Of_Ant[Main_Path_Node] = Index_Of_Sack  # Assign this sack to the ant's path.
                Main_Path_Node += 1  # Move to the next path node.
                Sum_Of_Weights += Money_Sack[Index_Of_Sack, 0]  # Add the weight of the selected sack to the total weight.
                Main_Candidates[Index_Of_Sack] = 0  # Remove the chosen sack from the list of candidates.
                Main_Candidates[Money_Sack[:, 0] > (Peak_Capacity - Sum_Of_Weights)] = 0  # Remove sacks that exceed the capacity.
                Sum_Of_Candidates_ = np.sum(Main_Candidates)  # Update sum of candidates.
            
            Each_Ant_Path.append(Path_Of_Ant[Path_Of_Ant > 0])  # Store the final path after removing unused elements (zeros).
            # Calculate fitness:
            Ant_Fitness[ant] = np.sum(Money_Sack[Each_Ant_Path[ant], 1])  # Fitness is the sum of values of the selected sacks.

        # If we have found a new global best solution, save it:
        Maximum_Generation = np.max(Ant_Fitness)  # Find the best fitness.
        Index_Number = np.argmax(Ant_Fitness)  # Find the index of the best fitness.

        if Main_generation == 0 or Maximum_Generation >= Peak_Value[0][Main_generation - 1]:  # Check if it's the best solution so far.
            Peak_Value[0][Main_generation] = Maximum_Generation  # Update global maximum value.
            Peak_Value[1] = Each_Ant_Path[Index_Number]  # Save the best path.
        else:
            Peak_Value[0][Main_generation] = Peak_Value[0][Main_generation - 1]  # Carry forward the previous best solution.

        # Dilute pheromones
        Pheromone_Variable *= Evaporation_Variable  # Evaporate pheromones.

        # Add pheromones
        Pheromone_Added = lambda Fit_Variable: Deposition_Variable * np.float32(Fit_Variable)  # Calculate pheromone addition based on fitness.
        
        # Find the n best fitness values
        Sorted_Vals = sorted([(fit, idx) for idx, fit in enumerate(Ant_Fitness)], key=lambda x: x[0])  # Sort ants based on fitness values.
        Given_Indexes = [idx for _, idx in Sorted_Vals[-AntPop:]]  # Get the indices of the top n ants.
        
        for i in Given_Indexes:  # Update pheromones for paths of the top n ants.
            Pheromone_Variable[Each_Ant_Path[i]] += Pheromone_Added(Ant_Fitness[i])

    return Peak_Value[0]  # Return the global maximum value after all generations.



import numpy as np

def File_Reader(file_name):
    with open(file_name, 'r') as file:
        # Read and parse maximum capacity from the first line
        try:
            first_line = file.readline().strip()
            Maximum_Capacity = int(first_line.split(':')[1].strip())
        except (IndexError, ValueError):
            raise ValueError("Error: Unable to read the van's maximum capacity.")
        
        # Calculate the number of sacks based on lines in the file
        lines = file.readlines()
        Number_Sack = (len(lines)) // 3
        
        # Initialize sacks array
        Sacks = np.zeros((Number_Sack, 2), dtype=np.float32)
        
        # Read sack details
        for i in range(Number_Sack):
            try:
                # Weight line
                weight_line = lines[3 * i + 1].strip()
                weight = float(weight_line.split(':')[1].strip())
                
                # Value line
                value_line = lines[3 * i + 2].strip()
                value = float(value_line.split(':')[1].strip())
                
                # Assign to Sacks array
                Sacks[i, 0] = weight #Assigns to one column
                Sacks[i, 1] = value # Asigns to other column
            except (IndexError, ValueError) as e:
                print(f"Warning: Skipping incomplete or misformatted sack data at lines {3 * i + 1} - {3 * i + 3}: {e}")
                continue
    
    return Sacks, Maximum_Capacity



def Random_Index_Generator(Candidate_Subj, Candidate_Summer):
    """Generates a random index based on pheromone distribution"""
    Random_Number = Candidate_Summer * np.random.rand() # Generated random number
    Sum_Candid = 0 # Initialized sum to 0
    
    for main_indexer, candidate in enumerate(Candidate_Subj):
        Sum_Candid += candidate #Increments to candidate function
        if Random_Number < Sum_Candid:
            return main_indexer
    return None


def Removing_Heavier_Sacks(candidates, weights, weightFree):
    """Removes sacks that are too heavy"""
    i = len(candidates) - 1  # Start from the last candidate.

    # Jump over zeroes in the end
    while candidates[i] == 0:
        i -= 1

    # Remove all sacks that are too heavy
    while i >= 0 and (candidates[i] == 0 or weights[i] > weightFree):
        candidates[i] = 0
        i -= 1
    return candidates



# Main script for solving Knapsack Problem

# Parameters of the test
Total_Generations = 10000 # Total number of iterations for the algorithm (generations).
Ant_Population = 20  # Population size: The number of ants in each generation.
Evaporation_Variable = 0.92  # Evaporation rate: Controls how much pheromone decays over time.
Deposition_Variable = 0.00001  # Pheromone deposition rate: The amount of pheromone deposited by ants after each move.
AntPop = 3  # Number of top ants used for pheromone deposition.
Iterator_Variable = 10  # Number of iterations to run the algorithm.

Maximum_Vector = np.zeros((Total_Generations, Iterator_Variable), dtype=np.uint16)  # Initialize matrix to store the global maximum values for each iteration.

start_time = time.time()  # Start timer for measuring performance.
for f in range(Iterator_Variable):  # Run multiple iterations.
    Maximum_Vector[:, f] = Main_Knap(Ant_Population, Evaporation_Variable, Deposition_Variable, Total_Generations, AntPop)  # Call the Knapsack function for each iteration.
end_time = time.time()  # Stop timer.

print(f"Execution Time: {end_time - start_time} seconds")  # Display execution time.

# Plot results
plt.figure()  # Create a figure for plotting.
plt.plot([1, Total_Generations], [4528, 4528], 'g')  # Plot a green line showing the maximum possible value (4528).
for f in range(Iterator_Variable):  # Loop through each iteration and plot the results.
    plt.plot(range(1, Maximum_Vector.shape[0] + 1), Maximum_Vector[:, f])  # Plot the global maximum value for each generation.

finalMax = np.max(Maximum_Vector[-1, :])  # Get the highest value from the final generation.
print(f"Final Max: {finalMax}")


