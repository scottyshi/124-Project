"""
    Imports
"""
import numpy as np
import operator
import time
import sys

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
def loadGenotypes(filename):
    with open(filename) as f:
        raw_data = f.readlines()
    data_out = np.array([[x.strip() for x in r.split(' ')] \
                         for r in raw_data]).T.astype(str)
    return data_out

"""
    Haplotype Phasing (Top Level)

    Goal: Top level function for haplotype phasing. Calls the "E" and "M"
        step until we reach convergence (no more changes to haplotype
        probabilities)

    Input: <np.array> -- (# individuals, genotype length)

    Output: {genotype: haplotype pair} -- mapping of genotype to its most
        probable haplotype pair.
"""
def phaseHaplotypes(genotypes, window=50, overlap=3):
    # TODO - magic number = 2x number of individuals
    decoded = ["" for i in range(100)]
    # First phase the initial chunk
    curr_chunk = genotypes[:,:window]
    curr_geno_haplos, curr_geno_to_haplo = \
        generateHaplotypes(curr_chunk, window)
    for i in range(4):
        EStep(curr_geno_to_haplo, curr_geno_haplos)
        MStep(curr_geno_to_haplo, curr_geno_haplos)
    cnt = 0
    for curr_g in curr_chunk:
        g_str = ''.join(curr_g)
        haplos = max(curr_geno_to_haplo[g_str].items(),\
                        key=operator.itemgetter(1))[0]
        decoded[cnt] += haplos[0]
        decoded[cnt+1] += haplos[1]
        cnt += 2
        
    # Then phase the overlap and next chunk
    for i in range(int(len(genotypes[0]) / window)-1):
        overlap_chunk = genotypes[:,\
                            (i+1)*window-overlap:(i+1)*window+overlap]
        overlap_geno_haplos, overlap_geno_to_haplo = \
            generateHaplotypes(overlap_chunk, 2*overlap)
        next_chunk = genotypes[:,(i+1)*window:(i+2)*window]
        next_geno_haplos, next_geno_to_haplo = \
            generateHaplotypes(next_chunk, window)

        # Run EM on a few iterations for each map
        for _ in range(2):
            EStep(overlap_geno_to_haplo, overlap_geno_haplos)
            MStep(overlap_geno_to_haplo, overlap_geno_haplos)
            EStep(next_geno_to_haplo, next_geno_haplos)
            MStep(next_geno_to_haplo, next_geno_haplos)
        
        # Decode this section while utilizing overlap
        decodeGenotypes(genotypes[:,i*window:(i+2)*window], window, i,\
                        overlap, overlap_geno_to_haplo, \
                        next_geno_to_haplo, decoded)    
    
    print("Finished slicing")
    # Randomly decode the remainder xD
    cnt = 0
    for geno in genotypes:
        geno = ''.join(geno)
        h1 = ""
        h2 = ""
        for snp in geno[int(len(geno)/window)*window:]:
            if snp == '0':
                h1 += '0'
                h2 += '0'
            elif snp == '2':
                h1 += '1'
                h2 += '1'
            else:
                if np.random.binomial(1,0.5) == 0:
                    h1 += '1'
                    h2 += '0'
                else:
                    h1 += '0'
                    h2 += '1'
        decoded[cnt] += h1
        decoded[cnt+1] += h2
        cnt += 2
    
    decoded = [[s for s in string] for string in decoded]
    
    # Make transpose, write to file
    outfile = "haplo_soln_" + str(window) + "_" + str(overlap) + ".txt"
    np.savetxt(outfile, np.array(decoded).astype(str).T, \
              delimiter=' ', newline='\r\n', fmt="%s")

def decodeGenotypes(geno_chunk, window, ind, overlap, \
                    overlap_geno_to_haplo, next_geno_to_haplo, decoded):
    cnt = 0
    for geno in geno_chunk:
        geno_str = ''.join(geno)
        
        g1_h1 = decoded[cnt][ind*window:(ind+1)*window]
        g1_h2 = decoded[cnt+1][ind*window:(ind+1)*window]
        
        overlap_str = geno_str[window-overlap:window+overlap]
        o_h1, o_h2 = max(overlap_geno_to_haplo[overlap_str].items(),\
                            key=operator.itemgetter(1))[0]
        
        g2 = geno_str[window:]
        g2_h1, g2_h2 = max(next_geno_to_haplo[g2].items(), \
                          key=operator.itemgetter(1))[0]
        
        perm1_h1 = g1_h1 + g2_h1
        perm1_h2 = g1_h2 + g2_h2
        perm2_h1 = g1_h1 + g2_h2
        perm2_h2 = g1_h2 + g2_h1

        # Find which ordering of g2's haplos is proper
        if (o_h1 == perm1_h1[window-overlap:window+overlap] \
            and o_h2 == perm1_h2[window-overlap:window+overlap]) \
            or (o_h2 == perm1_h2[window-overlap:window+overlap] \
             and o_h1 == perm1_h2[window-overlap:window+overlap]):
            decoded[cnt] += g2_h1
            decoded[cnt+1] += g2_h2
        else:
            decoded[cnt] += g2_h2
            decoded[cnt+1] += g2_h1
        
        cnt += 2    
    
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
    for haplo in possible_haplos:
        total_haplo_prob = 0
        for geno in geno_to_haplo:
            prob = getProbInGeno(geno_to_haplo[geno], haplo)
            total_haplo_prob += prob
        new_prob = total_haplo_prob / (2*num_genos)
        possible_haplos[haplo] = new_prob

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
        the list of our genotypes

    Input: <np.array> -- (# individuals, genotype length)

    Output:
        * {haplotype: haplotype probability} -- mapping of haplotype to
            it's initial probability = 1/n
        * {genotype: haplotypes} -- mapping of genotype to its associated
            haplotypes
"""
def generateHaplotypes(geno_chunk, window):
    geno_haplos, geno_to_haplos = {}, {}
    
    # Create initial set of haplotypes from first geno chunk
    sorted_geno_chunk = np.array(sorted(geno_chunk, \
                                key=lambda x: list(x).count('1')))    
    init_geno = ''.join(sorted_geno_chunk[0])
    haplos = []
    generateHaplotypesHelper(init_geno, \
                             ("",""), haplos)
    haplos = list(set(tuple(sorted(l)) for l in haplos))
    for h in haplos:
        geno_to_haplos[init_geno] = {h: -1}
        geno_haplos[h[0]] = 0
        geno_haplos[h[1]] = 0
        
    for geno in sorted_geno_chunk:
        geno_str = ''.join(geno)
        # If the genotype is not phasable, add the closest possible
        closest_phase = window
        # Keep list of new haplos that partially phase
        new_haplos = [] 
        # Potentially add closest haplo if no matches
        close_haplo = None
        for h in geno_haplos:
            h1, h2, num_switches = phasable(h, geno_str)
            if h1 > h2: # Keep ordering to help avoid duplicates
                h1, h2 = h2, h1
            if num_switches == 0:
                new_haplos += [(h1, h2)]
                closest_phase = -1 # No need to find closest phase
            else:
                if num_switches <= closest_phase:
                    closest_phase = num_switches
                    close_haplo = [(h1, h2)]
        if closest_phase != -1:
            if close_haplo is not None:
                new_haplos += close_haplo
            
        # Empty out list of new haplos into haplo and geno_to_haplo maps
        for new_h in new_haplos:
            if new_h[0] not in geno_haplos:
                geno_haplos[new_h[0]] = 0
            if new_h[1] not in geno_haplos:
                geno_haplos[new_h[1]] = 0
            if geno_str in geno_to_haplos:
                geno_to_haplos[geno_str][new_h] = -1
            else:
                geno_to_haplos[geno_str] = {new_h: -1}
    
    num_haplos = len(geno_haplos)
    for key in geno_haplos:
        geno_haplos[key] = float(1/num_haplos)
    
    return geno_haplos, geno_to_haplos
        
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
	window = 23
	overlap = 10
	haplo_file = sys.argv[1]
	start = time.time()
	genotypes = loadGenotypes(haplo_file)
	phaseHaplotypes(genotypes, window, overlap)
	end = time.time()
	print("Running time: ", (end - start))


if __name__ == '__main__':
	main()
