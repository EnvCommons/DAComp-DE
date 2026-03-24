from openreward.environments import Server

from dacomp_de import DACompDE

if __name__ == "__main__":
    server = Server([DACompDE])
    server.run()
