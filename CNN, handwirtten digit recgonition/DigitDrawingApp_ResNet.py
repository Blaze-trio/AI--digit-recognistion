import pygame
import numpy as np
import cv2
import tensorflow as tf
import sys
import os

class ResNetWrapper:
    """Wrapper to make ResNet compatible with DigitDrawingApp interface"""
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        
        self.model = tf.keras.models.load_model(model_path)
        print(f"ResNet model loaded from {model_path}")
    
    def predict_single(self, image_array):
        """
        Predict single image
        Args:
            image_array: 28x28 numpy array (normalized 0-1)
        Returns:
            predicted_digit, confidence, predictions_array
        """
        # Reshape for model input (batch_size, height, width, channels)
        image_batch = image_array.reshape(1, 28, 28, 1)
        
        # Get predictions
        predictions = self.model.predict(image_batch, verbose=0)[0]
        
        # Get predicted digit and confidence
        predicted_digit = np.argmax(predictions)
        confidence = predictions[predicted_digit]
        
        return predicted_digit, confidence, predictions

class DigitDrawingApp:
    def __init__(self, model_path="With-AI-Framework-Tensor-flow/resnet50_mnist_model.h5"):
        pygame.init()
        
        # Constants
        self.CANVAS_SIZE = 280  # 28x28 scaled up by 10
        self.WINDOW_WIDTH = 600
        self.WINDOW_HEIGHT = 400
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("ResNet Digit Recognition - Draw a digit (0-9)")
        
        self.canvas = pygame.Surface((self.CANVAS_SIZE, self.CANVAS_SIZE))
        self.canvas.fill(self.WHITE)

        # Load ResNet model
        try:
            self.model = ResNetWrapper(model_path)
            print("ResNet model loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please train the ResNet model first by running ResCNN.py")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        self.drawing = False
        self.brush_size = 15
        
        # Prediction variables
        self.current_prediction = None
        self.confidence = 0.0
        self.predictions_array = None
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Timing for predictions
        self.last_draw_time = 0
        self.prediction_delay = 500  # milliseconds

    def canvas_to_model_input(self):
        """Convert canvas to 28x28 grayscale array for model input"""
        canvas_array = pygame.surfarray.array3d(self.canvas)
        canvas_array = np.transpose(canvas_array, (1, 0, 2))  
        
        gray = cv2.cvtColor(canvas_array, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Invert colors (black drawing on white -> white on black for MNIST)
        inverted = 255 - resized
        
        # Normalize to 0-1 range
        normalized = inverted.astype(np.float32) / 255.0
        
        return normalized
    
    def predict_digit(self):
        """Get prediction from the ResNet model"""
        try:
            model_input = self.canvas_to_model_input()
            predicted_digit, confidence, predictions = self.model.predict_single(model_input)
            
            self.current_prediction = predicted_digit
            self.confidence = confidence
            self.predictions_array = predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            self.current_prediction = None
    
    def draw_on_canvas(self, pos):
        """Draw on the canvas at given position"""
        canvas_x = pos[0] - 50  
        canvas_y = pos[1] - 50  
        
        if 0 <= canvas_x < self.CANVAS_SIZE and 0 <= canvas_y < self.CANVAS_SIZE:
            pygame.draw.circle(self.canvas, self.BLACK, (canvas_x, canvas_y), self.brush_size)
            self.last_draw_time = pygame.time.get_ticks()
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.fill(self.WHITE)
        self.current_prediction = None
        self.confidence = 0.0
        self.predictions_array = None
    
    def draw_ui(self):
        """Draw the user interface"""
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Draw canvas border
        pygame.draw.rect(self.screen, self.BLACK, (48, 48, self.CANVAS_SIZE + 4, self.CANVAS_SIZE + 4), 2)
        
        # Draw canvas
        self.screen.blit(self.canvas, (50, 50))
        
        # Title
        title_text = self.font_large.render("ResNet Digit Recognition", True, self.BLACK)
        self.screen.blit(title_text, (350, 20))
        
        # Instructions
        instructions = [
            "Draw a digit (0-9) in the box",
            "Prediction updates automatically",
            "Press 'C' to clear",
            "Press 'Q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, self.BLACK)
            self.screen.blit(text, (350, 80 + i * 25))
        
        # Show predictions
        if self.current_prediction is not None:
            pred_text = self.font_large.render(f"Prediction: {self.current_prediction}", True, self.BLUE)
            self.screen.blit(pred_text, (350, 200))
            
            conf_text = self.font_medium.render(f"Confidence: {self.confidence:.2%}", True, self.BLUE)
            self.screen.blit(conf_text, (350, 240))
            
            # Show all predictions as bars
            if self.predictions_array is not None:
                bar_text = self.font_small.render("All Predictions:", True, self.BLACK)
                self.screen.blit(bar_text, (350, 280))
                
                for digit in range(10):
                    prob = self.predictions_array[digit]
                    bar_width = int(prob * 100) 
                    bar_height = 15
                    bar_y = 300 + digit * 20
                    
                    # Background bar
                    pygame.draw.rect(self.screen, self.GRAY, (350, bar_y, 100, bar_height))
                    
                    # Probability bar
                    color = self.GREEN if digit == self.current_prediction else self.RED
                    pygame.draw.rect(self.screen, color, (350, bar_y, bar_width, bar_height))
                    
                    # Text label
                    digit_text = self.font_small.render(f"{digit}: {prob:.1%}", True, self.BLACK)
                    self.screen.blit(digit_text, (460, bar_y))
        
        # Brush size indicator
        brush_text = self.font_small.render(f"Brush size: {self.brush_size} (use +/- to change)", True, self.BLACK)
        self.screen.blit(brush_text, (50, self.CANVAS_SIZE + 80))
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("ResNet Digit Drawing App Started!")
        print("Draw a digit in the white box. The ResNet model will predict what you drew.")
        print("Controls:")
        print("  - Mouse: Draw")
        print("  - C: Clear canvas")
        print("  - +/-: Change brush size")
        print("  - Q: Quit")
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_c:
                        self.clear_canvas()
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.brush_size = min(25, self.brush_size + 2)
                    elif event.key == pygame.K_MINUS:
                        self.brush_size = max(5, self.brush_size - 2)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  
                        self.drawing = True
                        self.draw_on_canvas(event.pos)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1: 
                        self.drawing = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.drawing:
                        self.draw_on_canvas(event.pos)
            
            # Auto-predict after drawing stops
            if (self.last_draw_time > 0 and 
                current_time - self.last_draw_time > self.prediction_delay):
                self.predict_digit()
                self.last_draw_time = 0 
            
            # Draw everything
            self.draw_ui()
            pygame.display.flip()
            clock.tick(60) 
        
        pygame.quit()
        print("App closed.")

def main():
    # Try to load ResNet model
    model_path = "With-AI-Framework-Tensor-flow/resnet50_mnist_model.h5"
    
    # Alternative paths to try
    alternative_paths = [
        "resnet50_mnist_model.h5",
        os.path.join("With-AI-Framework-Tensor-flow", "resnet50_mnist_model.h5"),
        os.path.join("..", "With-AI-Framework-Tensor-flow", "resnet50_mnist_model.h5")
    ]
    
    model_found = False
    for path in [model_path] + alternative_paths:
        if os.path.exists(path):
            model_path = path
            model_found = True
            break
    
    if not model_found:
        print("❌ ResNet model not found!")
        print("Please train the model first by running:")
        print("  1. cd 'With-AI-Framework-Tensor-flow'")
        print("  2. python ResCNN.py")
        return
    
    try:
        app = DigitDrawingApp(model_path)
        app.run()
    except Exception as e:
        print(f"Error starting app: {e}")

if __name__ == "__main__":
    main()