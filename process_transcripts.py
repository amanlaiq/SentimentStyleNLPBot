import string
import re
import csv
import nltk


# Function: process_transcripts(fname)
# fname: A string indicating a file name
# Returns: Nothing (writes output to file)

def process_transcripts(fname):
    f_in = open(fname, "r")
    f_out_all = open("all_{0}".format(fname), "w")
    f_out_chatbot = open("chatbot_{0}".format(fname), "w")
    f_out_user = open("user_{0}".format(fname), "w")

    interactor = None  

    for l in f_in:
        l = l.strip()

        if not l:
            continue

        if l.startswith("CHATBOT:"):
            interactor = "CHATBOT"
            continue  
        elif l.startswith("USER:"):
            interactor = "USER"
            continue  

        if interactor == "CHATBOT":
            chatbot_words = l  
            f_out_all.write(chatbot_words + "\n")
            f_out_chatbot.write(chatbot_words + "\n")
        elif interactor == "USER":
            user_words = l 
            f_out_all.write(user_words + "\n")
            f_out_user.write(user_words + "\n")

    f_in.close()
    f_out_all.close()
    f_out_chatbot.close()
    f_out_user.close()

    return

if __name__ == "__main__":
    fname = "2024-11-19_21-26-56.txt"
    process_transcripts(fname)