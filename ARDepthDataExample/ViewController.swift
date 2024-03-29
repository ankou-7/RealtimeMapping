/*
See LICENSE folder for this sample’s licensing information.

Abstract:
View controller that connects the host app renderer to a display.
*/

import UIKit
import Metal
import MetalKit
import ARKit

extension MTKView: RenderDestinationProvider {
}

class ViewController: UIViewController, MTKViewDelegate, ARSessionDelegate {
    
    var session: ARSession!
    var configuration = ARWorldTrackingConfiguration()
    var renderer: Renderer!
    var depthBuffer: CVPixelBuffer!
    var confidenceBuffer: CVPixelBuffer!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set this view controller as the session's delegate.
        session = ARSession()
        session.delegate = self
        
        // Set the view to use the default device.
        if let view = self.view as? MTKView {
            view.device = MTLCreateSystemDefaultDevice()
            view.backgroundColor = UIColor.clear //UIColor.blue
            view.delegate = self
            
            guard view.device != nil else {
                print("Metal is not supported on this device")
                return
            }
            
            print(view.bounds)
            
            // Configure the renderer to draw to the view.
            renderer = Renderer(session: session, metalDevice: view.device!, renderDestination: view)
            
            // Schedule the screen to be drawn for the first time.
            renderer.drawRectResized(size: view.bounds.size)
        }
        
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Enable the smoothed scene depth frame-semantic.
        configuration.frameSemantics = .smoothedSceneDepth
        
        configuration.environmentTexturing = .none
        
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            configuration.sceneReconstruction = .mesh//WithClassification
        }

        // Run the view's session.
        session.run(configuration)
        
        // The screen shouldn't dim during AR experiences.
        UIApplication.shared.isIdleTimerDisabled = true
    }
    
    // MARK: - MTKViewDelegate
    
    // Called whenever view changes orientation or size.
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Schedule the screen to be redrawn at the new size.
        renderer.drawRectResized(size: size)
    }
    
    // Implements the main rendering loop.
    func draw(in view: MTKView) {
        renderer.update()
    }
    
    // MARK: - ARSessionDelegate
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        // Present an error message to the user.
        guard error is ARError else { return }
        let errorWithInfo = error as NSError
        let messages = [
            errorWithInfo.localizedDescription,
            errorWithInfo.localizedFailureReason,
            errorWithInfo.localizedRecoverySuggestion
        ]
        let errorMessage = messages.compactMap({ $0 }).joined(separator: "\n")
        DispatchQueue.main.async {
            // Present an alert informing about the error that has occurred.
            let alertController = UIAlertController(title: "The AR session failed.", message: errorMessage, preferredStyle: .alert)
            let restartAction = UIAlertAction(title: "Restart Session", style: .default) { _ in
                alertController.dismiss(animated: true, completion: nil)
                self.session.run(self.configuration, options: .resetSceneReconstruction)
            }
            alertController.addAction(restartAction)
            self.present(alertController, animated: true, completion: nil)
        }
    }
    
    //var IDFlag = false
    
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for anchor in anchors {
            if anchor is ARMeshAnchor {
                renderer.meshFlag = true
            }
        }
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        var meshAnchors = [ARMeshAnchor]()
        print("update-------------------------------------------------------------------")
        for anchor in anchors {
            if let meshAnchor = anchor as? ARMeshAnchor {
                print(meshAnchor)
                meshAnchors.append(meshAnchor)
                //renderer.ID = meshAnchor.identifier

                //let n = renderer.anchorID[meshAnchor.identifier]!
                //renderer.meshAnchorsArray[n] = meshAnchor
            }
        }
        renderer.meshAnchors = meshAnchors
        print("--------------------------------------------------------------------------")
    }
                    

    // Auto-hide the home indicator to maximize immersion in AR experiences.
    override var prefersHomeIndicatorAutoHidden: Bool {
        return true
    }
    
    // Hide the status bar to maximize immersion in AR experiences.
    override var prefersStatusBarHidden: Bool {
        return true
    }
    
    // MARK: Toggle smoothing
    
    // Switch between temporally smoothed and single-frame scene depth.
    @IBAction func smoothingSwitchToggled(_ uiSwitch: UISwitch) {
        if uiSwitch.isOn {
            configuration.frameSemantics = .smoothedSceneDepth
        } else {
            configuration.frameSemantics = .sceneDepth
        }
        session.run(configuration)
    }
}
