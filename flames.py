from collections import Counter


FLAMES_DICTIONARY = {
    'F':'Friendship',
    'L':'Love',
    'A':'Affection',
    'M':'Marriage',
    'E':'Enemy',
    'S':'Sibling'
    }

def get_common_letters(string1, string2):
    '''Given two strings, remove common letters.

    Params
    ------
    string1, string2 : str

    
    Returns
    -------
    dict
        Dictionary of unique letters in both strings and their counts.
    '''
    # dict1 = Counter(string1)
    # dict2 = Counter(string2)

    # common1 = dict1 - dict2
    # common2 = dict2 - dict1

    # commonDict = common1 + common2

    #### if lahat ng common letters tatanggalin
    common1 = [char for char in string1 if char not in string2]
    common2 = [char for char in string2 if char not in string1]
    commonDict = common1 + common2
    commonDict = Counter(commonDict)  
    ####

    if len(commonDict) == 0:
        print('No letters in common')
        return None
    
    else:
        return commonDict

def get_flames_status(letter_list):
    '''Eliminate letters from FLAMES until one 
        is left given number of unique letters.

    Params
    ------
    letter_list : dict
        Dictionary of unique letters and their counts.


    Returns
    -------
    flames_status : str
        One of the letters in 'FLAMES'.


    '''
    flames_number = sum(letter_list.values())
    print('Remaining letters:', flames_number)
    flames_letters = [char for char in 'FLAMES']
    flames_index = (flames_number % len(flames_letters)) - 1

    while len(flames_letters) > 1 :
        # print('letter to remove: ',flames_letters[flames_index])
        flames_letters.pop(flames_index)

        temp_letters = [0 for i in range(len(flames_letters))]
        temp_letters = [flames_letters[n-(flames_index)-1] for n in range(len(flames_letters))]

        flames_letters = temp_letters
        flames_index = (flames_number % len(flames_letters)) - 1

    flames_status = flames_letters[0]
    return flames_status

if __name__ == '__main__':


    letters = get_common_letters('elisabeth','alexander')
    flames_status = get_flames_status(letters)
    print(FLAMES_DICTIONARY[flames_status])