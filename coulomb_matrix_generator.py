import numpy as np

class Molecule:
    # this class reads in molecular xyz files, uses the info to build a class object with positional attributes, and then performs several different functions for creating descriptors of the molecule for subsequent comparision with other molecules
    
    # this first __init__ function initialises the Molecule class object
    # it recieves a string of the name of the chemical xyz file as the arguement
    # * note that the xyz file must be in standard format and in the same directory as this .py script for it to be read by the program
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            content = list(file) # the content is stored as a list with each list member being one line from the xyz file
            
        self.natoms = int(content[0]) # the first line in the xyz file contains the number of atoms, so this is assigned to the natoms attribute
        
        atoms = [] # initialise an empty list
        for line in content[2:]: # now we want to store the coordinates, so information is only retrieved from the 3rd line onwards 
            element_symbol = line.split()[0] # the symbol is the first member of each list/line when they are split by spaces, so that is retrieved and stored as a variable
            atoms.append([element_symbol,np.array([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])]) # the atom list of lists is built up where each sub-list contains info of each atom, consisting of the atomic symbol and an array of the 3-dimensional coordinates 
            
        self.xyz = atoms # the atoms list of lists is assigned as an attribute to the Molecule class object
       
    
    def rawCM(self):
        
        # this function returns the Coulomb matrix of the Molecule object  
        
        atomic_number={
        " H ": 1 ," He ": 2 ," Li ": 3 ," Be ": 4 ," B ": 5 ," C ": 6 ," N ": 7 ," O ": 8 ," F ": 9 ," Ne ": 10 ,
        " Na ": 11 ," Mg ": 12 ," Al ": 13 ," Si ": 14 ," P ": 15 ," S ": 16 ," Cl ": 17 ," Ar ": 18 ," K ": 19 ,
        " Ca ": 20 ," Sc ": 21 ," Ti ": 22 ," V ": 23 ," Cr ": 24 ," Mn ": 25 ," Fe ": 26 ," Co ": 27 ," Ni ": 28 ,
        " Cu ": 29 ," Zn ": 30 ," Ga ": 31 ," Ge ": 32 ," As ": 33 ," Se ": 34 ," Br ": 35 ," Kr ": 36 ," Rb ": 37 ,
        " Sr ": 38 ," Y ": 39 ," Zr ": 40 ," Nb ": 41 ," Mo ": 42 ," Tc ": 43 ," Ru ": 44 ," Rh ": 45 ," Pd ": 46 ,
        " Ag ": 47 ," Cd ": 48 ," In ": 49 ," Sn ": 50 ," Sb ": 51 ," Te ": 52 ," I ": 53 ," Xe ": 54 ," Cs ": 55 ,
        " Ba ": 56 ," La ": 57 ," Ce ": 58 ," Pr ": 59 ," Nd ": 60 ," Pm ": 61 ," Sm ": 62 ," Eu ": 63 ," Gd ": 64 ,
        " Tb ": 65 ," Dy ": 66 , " Ho ": 67 ," Er ": 68 ," Tm ": 69 ," Yb ": 70 ," Lu ": 71 ," Hf ": 72 ," Ta ": 73 ,
        " W ": 74 ," Re ": 75 ," Os ": 76 ," Ir ": 77 ," Pt ": 78 ," Au ": 79 ," Hg ": 80 ," Tl ": 81 ," Pb ": 82 ,
        " Bi ": 83 , " Po ": 84 ," At ": 85 ," Rn ": 86 ," Fr ": 87 ," Ra ": 88 ," Ac ": 89 ," Th ": 90 ," Pa ": 91 ,
        " U ": 92 ," Np ": 93 ," Pu ": 94 ," Am ": 95 ," Cm ": 96 ," Bk ": 97 ," Cf ": 98 ," Es ": 99 ," Fm ": 100 ,
        " Md ": 101 ," No ": 102 ," Lr ": 103 ," Rf ": 104 ," Db ": 105 ," Sg ": 106 ," Bh ": 107 ," Hs ": 108 ,
        " Mt ": 109 ," Ds ": 110 ," Rg ": 111 ," Cn ": 112 } # start by defining a dictionary containing the atomic numbers of elements up to Cn (112)
        
        matrix = np.zeros([self.natoms,self.natoms]) # a 2D array (or matrix) of size N x N (where N is number of atoms in the molecule) consisting of zeros is initialised
        
        # this nested for loop iterates through each element/cell in the matrix 
        for i in range(self.natoms): # this line iterates through each row 
            for j in range(self.natoms): # this line iterates through each column
                if i == j: # the row and column indices are equal for diagonal elements in the matrix
                    elem = self.xyz[i][0] # isolate the atomic symbol from the xyz attribute (first element in the list of lists)
                    elem_str = str(" " + elem + " ") # put the string in the correct format for searching up in the dictionary (add spaces on each side)
                    num = atomic_number[elem_str] # the atomic number is looked up in the dictionary using the atomic symbol
                    matrix[i,j] = 0.5 * (num ** 2.4) # the calculation is then performed and that value is assigned to the diagonal grid element
                
                # and now for non-diagonal elements:
                else: 
                    elemi = self.xyz[i][0] # atomic symbol of atom 1 is retrieved as above
                    elemi_str = str(" " + elemi + " ") # string of atom 1 is put into correct form as above 
                    numi = atomic_number[elemi_str] # atomic number of atom 1 is searched up in dictionary
                    # the same is done for the second atom:
                    elemj = self.xyz[j][0]
                    elemj_str = str(" " + elemj + " ")
                    numj = atomic_number[elemj_str]
            
                    matrix[i,j] = numi * numj / np.linalg.norm(self.xyz[i][1] - self.xyz[j][1]) # the calculation is performed and value is assigned to the correctly indexed element in the matrix
                 
        return matrix # function returns the raw Coulomb matrix
    
    
    def eigenCM(self):
        
        # this function returns the eigenvalues of the raw Coulomb matrix
        
        eigs = np.linalg.eig(self.rawCM())[0] # the numpy eig function is used on the raw Coulomb matrix, which outputs both a list of the eigenvalues, in addition to the eigenvectors, which is why [0] is used to index and isolate just the eigenvalues
        sorted_eigs = np.sort(eigs)[::-1] # the eigenvalues are sorted into descending order (need to add the [::-1] as np.sort uses ascending order as default)
        return sorted_eigs # a 1D array of the sorted eigenvalues with size N (natoms) is returned
    
    
    def eigenCM_distance(self, vec):
        
        # this function returns the distance between two Molecules (Euclidian norm of the difference between the eigenvalue vectors of each molecule)
        # it recieves another Molecule object as an argument
        
        eig_vec1 = self.eigenCM() # the eigenvalue vectors are assigned to variables for the first molecule 
        eig_vec2 = vec.eigenCM() # and for the second molecule
        
        if len(eig_vec1) > len(eig_vec2): # checks if the first molecule is larger than the second (to account for different molecule sizes)
            len_diff = len(eig_vec1) - len(eig_vec2) # calculates the size difference 
            scaled_eig_vec2 = np.append(eig_vec2, [0] * len_diff) # the eigenvalue vector of the second molecule is appended with zeros to make it the same size as the first
            eig_diff = eig_vec1 - scaled_eig_vec2 # eigenvalue vector difference is calculated
            distance = np.linalg.norm(eig_diff) # the magnitude is calculated using numpy normalisation function
            
        elif len(eig_vec1) < len(eig_vec2): # checks if the second molecule is larger than the first then does same process
            len_diff = len(eig_vec2) - len(eig_vec1) # calculates size difference 
            scaled_eig_vec1 = np.append(eig_vec1, [0] * len_diff) # appends zeros to make vectors same size 
            eig_diff = scaled_eig_vec1 - eig_vec2 # vector difference is calculated 
            distance = np.linalg.norm(eig_diff) # magnitude is calculated 
            
        elif len(eig_vec1) == len(eig_vec2): # if two molecules have the same number of atoms
            eig_diff = eig_vec1 - eig_vec2 # just need to calculate difference (no need to append any zeros)
            distance = np.linalg.norm(eig_diff) # and then calculate the magnitude 
            
        return distance # the Euclidian norm is returned 
    
    
    def sortedCM(self):
        
        # this function returns the sorted Coulomb matrix of the Molecule object
        
        norm_vals_list = [] # an empty list for storing normalised row values is initialised 
        
        for line in self.rawCM(): # loop to iterate through each row in the raw Coulomb matrix
            norm_vals_list.append(np.linalg.norm(line)) # the Euclidian norm of each row is appended to the list in the same order as the atoms in the xyz file
            
        norm_vals_array = np.array(norm_vals_list) # the list is transformed into a 1D numpy array due to the increased functionality this adds (in terms of operations that can be performed)
        sort_order = norm_vals_array.argsort()[::-1] # the norm values are run through the argsort function, which returns a 1D array containing the indices of each list member sorted into descending order (not the actual norm values sorted into descending order)

        raw_CM = self.rawCM() # the raw Coulomb matrix is assigned to a variable
        sorted_CM = np.zeros([self.natoms,self.natoms]) # an empty sorted Coloumb matrix is initialised containing zeros of size N x N
    
        # nested loop to iterate through each cell in the raw Coulomb matrix (could iterate through len(sorted_CM) instead which would give same outcome as this simply defines the range of the loop to iterate through, which equals N / the matrix size for both)
        for i in range(len(raw_CM)): # iterates through each row
            for j in range(len(raw_CM)): # iterates through each column for each row
                sorted_CM[i,j] = raw_CM[sort_order[i],sort_order[j]] # values are assigned to the sorted CM grid based on the sorting order 
                # the new sorted values are obtained by indexing the raw Coulomb matrix with indices from the sort_order array (which contains the order that the rows and columns need to be sorted into) and then assigning the values to their sorted positions in the grid
                # here, both the rows and columns are sorted into the same atom order based on the order of descending norm values of each row
                # this ensures that the diagonal values are still in the same form (where row and col are same element)
                
        return sorted_CM # a sorted Coulomb matrix is returned
    
    
    def sortedCM_distance(self, vec):
        
        # this function calculates the Frobenius norm of the difference between the sorted Coulomb matrices of two molecules   
        # it recieves another Molecule object as an argument 
        
        sorted_CM1 = self.sortedCM() # the sorted Coulomb matrix of molecule 1 is assigned to a variable
        sorted_CM2 = vec.sortedCM() # and the sorted Coulomb matrix of molecule 2 is assigned to another variable 
        
        if self.natoms > vec.natoms: # checks if molecule 1 consists of more atoms than molecule 2
            atom_diff = self.natoms - vec.natoms # works out the size difference 
            scaled_sorted_CM2 = np.pad(sorted_CM2, [(0, atom_diff), (0, atom_diff)], mode='constant', constant_values=0) # the sorted Coulomb matrix of the smaller molecule is padded with zeros (in both dimensions) to make the number of rows and columns of the two matrices equal
            CM_diff = sorted_CM1 - scaled_sorted_CM2 # the difference between the sorted Coulomb matrices is calculated
            distance = np.linalg.norm(CM_diff, ord='fro') # the Frobenius norm is then calculated using the matrix difference 
            
        elif self.natoms < vec.natoms: # checks if molecule 2 is larger than molecule 1 then does the same process
            atom_diff = vec.natoms - self.natoms # calculate size difference 
            scaled_sorted_CM1 = np.pad(sorted_CM1, [(0, atom_diff), (0, atom_diff)], mode='constant', constant_values=0) # pad smaller matrix with sufficient zeros
            CM_diff = scaled_sorted_CM1 - sorted_CM2 # calculate matrix difference 
            distance = np.linalg.norm(CM_diff, ord='fro') # the Frobenius norm is calculated
            
        else: # when the molecules have the same number of atoms
            CM_diff = sorted_CM1 - sorted_CM2 # calculate matrix difference 
            distance = np.linalg.norm(CM_diff, ord='fro') # then calculate the Frobenius norm
            
        return distance # finally, the distance (i.e. Frobenius norm) is returned 
