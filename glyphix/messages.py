import pyfiglet

def generate_welcome_ascii_art(message="Welcome", font="slant"):
    """
    Generates ASCII art for the given message using a specified Figlet font.
    
    Parameters:
    - message: The message to convert into ASCII art.
    - font: The Figlet font to use for the ASCII art.
    """
    ascii_art = pyfiglet.figlet_format(message, font=font,width=120)
    print(ascii_art)



def display_goodbye_message():
    goodbye_messages = [
        "🚀 Journey well! Remember, every line of code brings you closer to greatness.",
        "💡 You've illuminated the path of innovation. Until next time, keep shining!",
        "🌟 You're leaving a star in the digital cosmos. Don't be a stranger to this universe!",
        "🌱 From tiny seeds grow mighty trees. Your progress is impressive! Keep growing.",
        "📘 You've added a valuable chapter to your book of coding adventures. Looking forward to your next story!",
        "🌉 You're building bridges to the future, one line of code at a time. Safe travels!",
        "🚀 Launching you back to reality with new knowledge and insights. May your code be bug-free and your ideas limitless.",
        "💖 Coding is like heartbeats for the digital world, and yours skipped no beat. Farewell, until our paths cross again!",
        "🎨 You've painted your ideas into reality. Take a bow, artist of the algorithm!",
        "🎓 Today, you learned and grew. Tomorrow, you conquer. Keep pushing the boundaries!",
    ]

    from random import choice
    print(choice(goodbye_messages))

