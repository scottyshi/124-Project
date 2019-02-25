"""
    Imports
"""
import numpy as np
import operator
import time

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
def phaseHaplotypes(genotypes, window=100):
    if window > len(genotypes[0]):
        print("Please choose a window smaller than genotype length")
        return

    STOPPING_CRITERION = 0.01

    possible_haplos, geno_to_haplo = generateHaplotypes(genotypes, window)

    # Run EM until there's little average change in the probabilities
    for i in range(3):
        print("Running EM iteration", i)
        EStep(geno_to_haplo, possible_haplos)
        avg_change = MStep(geno_to_haplo, possible_haplos)
        print("Average change this iteration: ", avg_change)
        # if avg_change < STOPPING_CRITERION:
        #     break

    print("Decoding genotype now")
    decodeGenotypes(genotypes, geno_to_haplo, window)

"""
    After running EM algorithm, decode the most probable haplotypes
"""
def decodeGenotypes(genotypes, geno_to_haplo, window):
    decoded = []
    for geno in genotypes:
        geno = ''.join(geno)
        h1 = ""
        h2 = ""
        for i in range(int(len(geno) / window)):
            geno_slice = geno[i*window:(i+1)*window]
            haplos = max(geno_to_haplo[geno_slice].items(), \
                         key=operator.itemgetter(1))[0]
            h1 += haplos[0]
            h2 += haplos[1]

        # Decode the rest of the genotype
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

        # Add to decoded list, convert string to list
        decoded.append([snp for snp in h1])
        decoded.append([snp for snp in h2])

    # Make transpose, write to file
    np.savetxt("haplo_soln.txt", np.array(decoded).astype(str).T, \
              delimiter=' ', newline='\r\n', fmt="%s")
"""
    Helper Function to Perform the E-Step
"""
def EStep(geno_to_haplo, possible_haplos):
    print("Running E Step")
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
    print("Running M Step")
    num_genos = len(geno_to_haplo)
    avg_change = 0
    print(len(possible_haplos))
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
        the list of our genotypes

    Input: <np.array> -- (# individuals, genotype length)

    Output:
        * {haplotype: haplotype probability} -- mapping of haplotype to
            it's initial probability = 1/n
        * {genotype: haplotypes} -- mapping of genotype to its associated
            haplotypes
"""
def generateHaplotypes(genotypes, window):
    print("Populating list of all haplotypes and genotypes")
    possible_haplos = {}
    geno_to_haplos = {}
    visited_geno_slices = set()
    for geno in genotypes:
        # Convert to string for easier hashing
        geno = ''.join(geno)
        # Slice the geno into the window sizes
        for i in range(int(len(geno) / window)):
            geno_slice = geno[i*window:(i+1)*window]

            # Skip phasing for
            if geno_slice in visited_geno_slices:
                continue
            else:
                visited_geno_slices.add(geno_slice)

            # Generate map of genotype to its haplotype pairs
            haplos = []
            generateHaplotypesHelper(geno_slice, ("",""), haplos)
            # Remove mirrored tuples
            haplos = list(set(tuple(sorted(l)) for l in haplos))
            for h in haplos:
                if geno_slice in geno_to_haplos:
                    geno_to_haplos[geno_slice][h] = -1
                else:
                    geno_to_haplos[geno_slice] = {h: -1}


            # Add to list of known haplotypes
            for hap in haplos:
                if hap[0] not in possible_haplos:
                    possible_haplos[hap[0]] = 0
                if hap[1] not in possible_haplos:
                    possible_haplos[hap[1]] = 0


    num_haplos = len(possible_haplos)
    for key in possible_haplos:
        possible_haplos[key] = float(1/num_haplos)
    return possible_haplos, geno_to_haplos

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
	start = time.time()
	genotypes = loadGenotypes("example_data_1.txt")
	phaseHaplotypes(genotypes, 15)
	end = time.time()
	print(end - start)

if __name__ == '__main__':
	main()
