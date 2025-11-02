class Titlecard:

    def print_titlecard(self):
        msg1 = "Welcome to SHAMPOO"
        msg2 = "StocHAstic Monomer PrOcessOr"
        msg3 = "Dynamical, collisional and ice processing since 2022"
        maxLen = len(msg3)

        msgList = [msg1, msg2, msg3]

        print(" " + "-" * (maxLen + 20) + " ")
        print("|" + " " * (maxLen + 20) + "|")
        for msg in msgList:
            msgLen = len(msg)
            space = " " * (int((maxLen - msgLen) / 2) + 10)
            print("|" + space + msg + space + "|")
            print("|" + " " * (maxLen + 20) + "|")

        print(" " + "-" * (maxLen + 20) + " ")