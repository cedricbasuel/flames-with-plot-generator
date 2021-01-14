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
        return 0
    
    else:
        return commonDict



def get_flames_status(letter_list):
    flames_number = sum(letter_list.values())
    print('Remaining letters:', flames_number)
    flames_letters = [char for char in 'FLAMES']
    flames_index = (flames_number % len(flames_letters)) - 1

    while len(flames_letters) > 1 :
        # print(flames_index)
        print('letter to remove: ',flames_letters[flames_index])
        flames_letters.pop(flames_index)

        temp_letters = [0 for i in range(len(flames_letters))]
        temp_letters = [flames_letters[n-(flames_index)-1] for n in range(len(flames_letters))]
        # print(temp_letters)

        flames_letters = temp_letters
        flames_index = (flames_number % len(flames_letters)) - 1

    return flames_letters[0]



letters = get_common_letters('elisabeth','alexander')
flames_status = get_flames_status(letters)
print(FLAMES_DICTIONARY[flames_status])