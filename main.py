from GestureDetector import FingerControl

if __name__ == "__main__":
    movements = {
        "up": "w",
        "down": "s",
        "left": "a",
        "right": "d"
    }

    app = FingerControl(x0=300, y0=900, x1=300, y1=900, movements=movements)

    app.start()
