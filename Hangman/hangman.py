import random

class Hangman:
    # This function will automatically end the game by itself
    def print_word(self):
        remaining = len(self.word)
        for c in self.word:
            if self.guessed.find(c) > -1:
                print(self.GREEN + c + " " + self.WHITE, end="")
                remaining -= 1
            else:
                print(self.GREEN + "_ " + self.WHITE, end="")
        if remaining == 0:
            print(f"{self.YELLOW}\nCongratulations! You correctly guessed the word! Quitting now.{self.WHITE}")
            exit(0)
        elif self.attempts == 0:
            print(f"{self.YELLOW}\nYou ran out of attempts! The word I chose was: {self.word} {self.WHITE}")
            exit(0)
        else:
            print(f"{self.WHITE}(Remaining attempts: {self.attempts})")

    def start_game(self):
        while 1:
            self.print_word()
            while 1:
                letter = input(f"{self.YELLOW}Please enter a single letter you'd like to insert: {self.WHITE}")
                letter.lower()
                if len(letter) != 1 or letter.isalpha() == False:
                    print(f"{self.RED}Illegal input! Please choose a [SINGLE] [LETTER]!{self.WHITE}")
                elif self.guessed.find(letter) > -1:
                    print(f"{self.RED}You've already used this letter! Guess again!{self.WHITE}")
                else:
                    self.guessed += letter
                    break
            if self.word.find(letter) == -1:
                print(f"{self.RED}Sorry, that letter doesn't exist in my word!{self.WHITE}")
                self.attempts -= 1

    def choose_word(self):
        try:
            with open("words.txt", "r") as file:
                l = random.randint(1, 150)
                content = file.read()
                line = 1
                for c in content:
                    # found the randomly chosen line
                    if l == line and c != "\n":
                        self.word += c.lower()
                    elif c == "\n":
                        line += 1
                    if line > l:
                        break
                self.word.strip()
                file.close()
        except FileNotFoundError:
            print(self.RED + "Error encountered while preparing the game. 'Words.txt' is missing!" + self.WHITE)
            exit(1)

    def __init__(self):
        self.word = ""
        self.guessed = ""
        self.attempts = 7
        # LIST OF USABLE COLOR-CODES
        self.RED = "\033[0;31m"
        self.GREEN = "\033[0;32m"
        self.YELLOW = "\033[1;33m"
        self.WHITE = "\033[1;37m"
        # SET UP GAME
        self.choose_word()
        self.start_game()

if __name__ == "__main__":
    print("Welcome to Hangman! Can you guess the word which I'll choose?")
    hangman = Hangman()