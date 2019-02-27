"""
    Imports
"""
import numpy as np
import operator

"""
    Dataset Loading
    
    Goal: Load the dataset and transform into the following format...
    
    Input: <str> -- filename of genotype data
        - (39267,50) 
            * 39267 SNPs, 50 individuals
        - SNP positions separated by row
        - Each column is an individual
        
    Output: <np.array> -- (50,39267) 
        - Will use row index as unique identifier for person
"""
def loadGenotypes(geno_filename, positions_filename):
    with open(geno_filename) as f:
        raw_data = f.readlines()
    data_out = np.array([[x.strip() for x in r.split(' ')] \
                         for r in raw_data]).T.astype(str)
    
    with open(positions_filename) as f:
        geno_pos = f.readlines()
    geno_out = np.array([r.strip() for r in geno_pos]).astype(int)
    return data_out, geno_out
    
"""
    Haplotype Phasing (Top Level)
    
    Goal: Top level function for haplotype phasing. Calls the "E" and "M"
        step until we reach convergence (no more changes to haplotype
        probabilities)
        
    Input: <np.array> -- (# individuals, genotype length)
    
    Output: {genotype: haplotype pair} -- mapping of genotype to its most
        probable haplotype pair. 
"""
def phaseHaplotypes(genotypes, geno_pos, window=125):
    # A few hundred thousand base pairs would be reasonable to assume
    #  the base pairs are in LD with each other
    ld_distance = 200000
    
    ld_indicies = getGenoIndices(genotypes, ld_distance, geno_pos)
    ld_indicies += [len(genotypes[0])]
    prev_ld_start = 0
    decoded = [[] for i in range(100)]
    for ld in ld_indicies:
        # Consider slices of the genotype one LD block at a time
        geno_ld_slice = genotypes[:, prev_ld_start:ld]
        # print("prev_ld_start: " + str(prev_ld_start) + " to " + str(ld))
        possible_haplos, geno_to_haplo, possible_haplos_r, \
            geno_to_haplo_r = generateHaplotypes(geno_ld_slice, window)
#         print("geno_slice", geno_ld_slice)
#         print("possible_haplos", possible_haplos)
#         print("geno_to_haplo", geno_to_haplo)
#         print("remainder", possible_haplos_r)
        # Run one iteration of EM
        EStep(geno_to_haplo, possible_haplos)
        avg_change = MStep(geno_to_haplo, possible_haplos)
        if len(possible_haplos_r) != 0:
            EStep(geno_to_haplo_r, possible_haplos_r)
            avg_change_r = MStep(geno_to_haplo_r, possible_haplos_r)
            
        # Start decoding
        individual_cnt = 0
        for geno_individual_slice in geno_ld_slice:
            h1 = ""
            h2 = ""
            geno = ''.join(geno_individual_slice)

            for i in range(int(len(geno_individual_slice) / window)):
                geno_slice = geno[i*window:(i+1)*window]
                haplos = max(geno_to_haplo[geno_slice].items(), \
                            key=operator.itemgetter(1))[0]
                h1 += haplos[0]
                h2 += haplos[1]
            if len(geno_to_haplo_r) != 0:
                geno_slice = geno[int(len(geno)/window)*window:]
                haplos = max(geno_to_haplo_r[geno_slice].items(),\
                            key=operator.itemgetter(1))[0]
                h1 += haplos[0]
                h2 += haplos[1]
            else:
                geno_slice = geno
                haplos = max(geno_to_haplo[geno_slice].items(), \
                            key=operator.itemgetter(1))[0]
                h1 += haplos[0]
                h2 += haplos[1]
            decoded[individual_cnt] += [snp for snp in h1]
            decoded[individual_cnt+1] += [snp for snp in h2]
            individual_cnt += 2
        
        prev_ld_start = ld
    print(np.array(decoded).shape)
    # Make transpose, write to file
    np.savetxt("haplo_clarks_soln.txt", np.array(decoded).astype(str).T, \
              delimiter=' ', newline='\r\n', fmt="%s")

"""
    Using the genotype positions, determine slices of the genotypes
        that are in LD with each other.
"""
def getGenoIndices(genotypes, ld_distance, geno_pos):
    ld_indicies = []
    # start_pos denotes the start of a new genotype slice
    #  we will consider the slice to be correlated among individuals
    start_pos = geno_pos[0]
    for i in range(len(geno_pos)):
        if geno_pos[i] - start_pos > ld_distance:
            ld_indicies.append(i)
            start_pos = geno_pos[i]
    return ld_indicies
    
"""
    Helper Function to Perform the E-Step
"""
def EStep(geno_to_haplo, possible_haplos):
    for geno in geno_to_haplo:
        normalized_geno_sum = 0
        # Calculate haplotype probabiliies based on curr guesses
        for haplo in geno_to_haplo[geno]:
            haplo_prob = possible_haplos[haplo[0]] * \
                                possible_haplos[haplo[1]]
            normalized_geno_sum += haplo_prob
            geno_to_haplo[geno][haplo] = haplo_prob
            
        # Normalize probabilities per genotype
        for haplo in geno_to_haplo[geno]:
            geno_to_haplo[geno][haplo] /= normalized_geno_sum
    
"""
    Helper Function to Perform the M-Step
"""
def MStep(geno_to_haplo, possible_haplos):
    num_genos = len(geno_to_haplo)
    avg_change = 0
    for haplo in possible_haplos:
        total_haplo_prob = 0
        for geno in geno_to_haplo:
            prob = getProbInGeno(geno_to_haplo[geno], haplo)
            total_haplo_prob += prob
        new_prob = total_haplo_prob / (2*num_genos)
        avg_change += abs(new_prob - possible_haplos[haplo])
        possible_haplos[haplo] = new_prob
        
    return avg_change / len(possible_haplos)
        
"""
    M-Step helper to retrieve haplotype prob within a genotype
"""
def getProbInGeno(haplo_list, haplo):
    for haplo_pair in haplo_list:
        h1, h2 = haplo_pair
        if h1 == haplo or h2 == haplo:
            return haplo_list[haplo_pair]
    return 0

"""
    Generate Haplotypes
    
    Goal: Help generate the list of haplotypes that can be produced from
        the list of our genotypes. To generate the list, I use pure
        parsimony, starting with the genotype that has the lowest amount
        
    Input: <np.array> -- (# individuals, genotype length)
    
    Output: 
        * {haplotype: haplotype probability} -- mapping of haplotype to
            it's initial probability = 1/n
        * {genotype: haplotypes} -- mapping of genotype to its associated
            haplotypes
"""
def generateHaplotypes(geno_ld_slice, window):
    possible_haplos = {}
    geno_to_haplos = {}
    
    # If window doesn't nicely divide up the ld_slice, also keep track
    #  of the remainders
    possible_haplos_r = {}
    geno_to_haplos_r = {}
    
    if len(geno_ld_slice[0]) <= window:
        sorted_geno_ld_slice = np.array(sorted(geno_ld_slice, \
                                    key=lambda x: list(x).count('1')))
        clarks(sorted_geno_ld_slice, possible_haplos, geno_to_haplos)
    else:
        # Slice the geno into the window sizes if ld slice too large
        for i in range(int(len(geno_ld_slice[0]) / window)):
            geno_slice = geno_ld_slice[:,i*window:(i+1)*window]
            sorted_geno_ld_slice = np.array(sorted(geno_slice, \
                                        key=lambda x: list(x).count('1')))
            clarks(sorted_geno_ld_slice, possible_haplos, geno_to_haplos)
        
        # Handle the remainder as well
        remainder_geno = geno_ld_slice[:, \
                        int(len(geno_ld_slice[0])/window)*window:]
        sorted_remainder_slice = np.array(sorted(remainder_geno,\
                                        key=lambda x: list(x).count('1')))
        clarks(sorted_remainder_slice, possible_haplos_r, geno_to_haplos_r)
        
    num_haplos = len(possible_haplos)
    for key in possible_haplos:
        possible_haplos[key] = float(1/num_haplos)
        
    if len(possible_haplos_r) != 0:
        num_haplos = len(possible_haplos_r)
        for key in possible_haplos_r:
            possible_haplos_r[key] = float(1/num_haplos)
        
    return possible_haplos, geno_to_haplos, \
            possible_haplos_r, geno_to_haplos_r
        
"""
    Clarks Algorithm
    
    Goal: Start with the first geno (sorted based on least # 1's) and 
        populate mapping of genotype to haplotypes and the haplotype
        probability map. 
"""
def clarks(geno_ld_slice, possible_haplos, geno_to_haplos):
    
    init_geno = ''.join(geno_ld_slice[0])
    haplos = []
    
    # Recursively generate initial list of known haplotypes
    generateHaplotypesHelper(init_geno, ("",""), haplos)
    haplos = list(set(tuple(sorted(l)) for l in haplos))
    for h in haplos:
        geno_to_haplos[init_geno] = {h: -1}
        possible_haplos[h[0]] = 0
        possible_haplos[h[1]] = 0
        
    for geno in geno_ld_slice:
        # Iterate through the known haplotypes, try to phase the genotype
        geno_str = ''.join(geno)
        closest_phase = 100
        closest_h1 = ""
        closest_h2 = ""
        for h in possible_haplos:
            h1, h2, num_switches = phasable(h, geno_str)
            if num_switches == 0:
                closest_h1 = h1
                closest_h2 = h2
                break
            else:
                if num_switches < closest_phase:
                    closest_phase = num_switches
                    closest_h1 = h1
                    closest_h2 = h2
        # Maintain ordering to prevent duplicates
        if closest_h1 > closest_h2:
            closest_h1, closest_h2 = closest_h2, closest_h1
            
        # Add the new haplotypes
        if closest_h1 not in possible_haplos:
            possible_haplos[closest_h1] = 0
        if closest_h2 not in possible_haplos:
            possible_haplos[closest_h2] = 0
            
        # Create the new genotype mapping
        if geno_str in geno_to_haplos:
            geno_to_haplos[geno_str][(closest_h1,closest_h2)] = -1
        else:
            geno_to_haplos[geno_str] = {(closest_h1,closest_h2): -1}
                
    
"""
    Phasable checks if a haplotype phasing is valid for a given genotype
    
    Returns h1, the other half, and 0 if it's phasable
    
    Returns h1, the other half, and and min number of switches if it's 
        not phasable
"""
def phasable(h1, geno):
    g_i = np.array([int(i) for i in geno])
    h_i = np.array([int(i) for i in h1])
    
    num_switches = list(g_i-h_i).count(-1) + list(g_i-h_i).count(2)
    
    if num_switches > 0:
        h_i = [0 if x == -1 else x for x in list(g_i-h_i)]
        h_i = np.array([1 if x == 2 else x for x in h_i])
        h1 = ''.join(h_i.astype(str))
    h2 = ''.join((g_i - h_i).astype(str))
    
    return h1, h2, num_switches
        
    
"""
    Generate Haplotypes Helper
    
    Goal: Given a single genotype, recursively create a list of all the
        possible haplotypes by looking at the genotype SNP by SNP. 
        Everytime we encounter a '1' in the genotype, make a recursive
        branch.
        
    Input: <np.array> (genotype length,)
    
    Output: <np.array> (# haplotypes generated, haplotype length)
"""
def generateHaplotypesHelper(geno, curr_haplo, haplos):
    if len(geno) == 0: 
        haplos.append(curr_haplo)
        return
    
    # Consider the next genotype SNP
    if geno[0] == '0':
        generateHaplotypesHelper(geno[1:], \
                                 ((curr_haplo[0] + '0'), \
                                 (curr_haplo[1] + '0')), \
                                 haplos)
    elif geno[0] ==  '2':
        generateHaplotypesHelper(geno[1:], \
                                 (curr_haplo[0] + '1', \
                                 curr_haplo[1] + '1'), \
                                 haplos)
    else:
        generateHaplotypesHelper(geno[1:], \
                                 (curr_haplo[0] + '0', \
                                 curr_haplo[1] + '1'), \
                                 haplos)
        generateHaplotypesHelper(geno[1:], \
                                 (curr_haplo[0] + '1', \
                                 curr_haplo[1] + '0'), \
                                 haplos)
    
def main():
    """
        MAIN CODE
    """
    import time

    start = time.time()
    genotypes, geno_pos = loadGenotypes("example_data_1.txt", \
                                        "example_data_1_geno_positions.txt")
    phaseHaplotypes(genotypes, geno_pos, 75)
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()