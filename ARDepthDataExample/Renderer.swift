/*
 See LICENSE folder for this sample’s licensing information.
 
 Abstract:
 The host app renderer.
 */

import Foundation
import Metal
import MetalKit
import ARKit
import MetalPerformanceShaders

protocol RenderDestinationProvider {
    var currentRenderPassDescriptor: MTLRenderPassDescriptor? { get }
    var currentDrawable: CAMetalDrawable? { get }
    var colorPixelFormat: MTLPixelFormat { get set }
    var sampleCount: Int { get set }
}

// The max number of command buffers in flight.
let kMaxBuffersInFlight: Int = 3

// Vertex data for an image plane.
let kImagePlaneVertexData: [Float] = [
    -1.0, -1.0, 0.0, 1.0,
     1.0, -1.0, 1.0, 1.0,
     -1.0, 1.0, 0.0, 0.0,
     1.0, 1.0, 1.0, 0.0
]

class Renderer {
    let session: ARSession
    let device: MTLDevice
    let inFlightSemaphore = DispatchSemaphore(value: kMaxBuffersInFlight)
    var renderDestination: RenderDestinationProvider
    
    // Metal objects.
    var commandQueue: MTLCommandQueue!
    
    // An object that holds vertex information for source and destination rendering.
    var imagePlaneVertexBuffer: MTLBuffer!
    
    // An object that defines the Metal shaders that render the camera image and fog.
    var fogPipelineState: MTLRenderPipelineState!
    var meshPipelineState: MTLRenderPipelineState!
    var calcuPipelineState: MTLRenderPipelineState!
    
    // Textures used to transfer the current camera image to the GPU for rendering.
    var cameraImageTextureY: CVMetalTexture?
    var cameraImageTextureCbCr: CVMetalTexture?
    
    // A texture used to store depth information from the current frame.
    var depthTexture: CVMetalTexture?
    
    // A texture used to pass confidence information to the GPU for fog rendering.
    var confidenceTexture: CVMetalTexture?
    
    // A texture of the blurred depth data to pass to the GPU for fog rendering.
    var filteredDepthTexture: MTLTexture!
    // A filter used to blur the depth data for rendering fog.
    var blurFilter: MPSImageGaussianBlur?
    
    // Captured image texture cache.
    var cameraImageTextureCache: CVMetalTextureCache!
    
    // The current viewport size.
    var viewportSize: CGSize = CGSize()
    
    // Flag for viewport size changes.
    var viewportSizeDidChange: Bool = false
    
    
    var meshFlag = false
    var anchor: ARMeshAnchor!
    var ID: UUID!
    var anchorNum : Int!
    
    var anchorUniforms: AnchorUniforms!
    var anchorUniformsBuffer: MTLBuffer!
    var viewProjectionMatrix: float4x4!
    
    private let relaxedStencilState: MTLDepthStencilState
    
    var kernelPipeline: MTLComputePipelineState!
    var update_kernelPipeline: MTLComputePipelineState!
    
    var anchorID = Dictionary<UUID, Int>() //ARMeshAnchorのidと追加番号
    var perFaceCount: [Int] = []
    var perFaceIndex: [Int] = [0]
    var sumCount: Int = 0
    
    var pre_vertexUniformsBuffer: MTLBuffer! //MetalBuffer<vertexUniforms>!
    private var vertexUniformsBuffer: MTLBuffer! //MetalBuffer<vertexUniforms>
    var calcuUniforms: MTLBuffer!
    
    // Initialize a renderer by setting up the AR session, GPU, and screen backing-store.
    init(session: ARSession, metalDevice device: MTLDevice, renderDestination: RenderDestinationProvider) {
        self.session = session
        self.device = device
        self.renderDestination = renderDestination
        
        vertexUniformsBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride * 99_999_999, options: []) //.init(device: device, count: 99_999_999, index: 8)
        let relaxedStateDescriptor = MTLDepthStencilDescriptor()
        relaxedStencilState = device.makeDepthStencilState(descriptor: relaxedStateDescriptor)!
        
        // Perform one-time setup of the Metal objects.
        loadMetal()
    }
    
    // Schedule a draw to happen at a new size.
    func drawRectResized(size: CGSize) {
        viewportSize = size
        viewportSizeDidChange = true
    }
    
    func update() {
        // Wait to ensure only kMaxBuffersInFlight are getting proccessed by any stage in the Metal
        // pipeline (App, Metal, Drivers, GPU, etc).
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        // Create a new command buffer for each renderpass to the current drawable.
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            commandBuffer.label = "MyCommand"
            
            // Add completion hander which signal _inFlightSemaphore when Metal and the GPU has fully
            // finished proccssing the commands we're encoding this frame.  This indicates when the
            // dynamic buffers, that we're writing to this frame, will no longer be needed by Metal
            // and the GPU.
            commandBuffer.addCompletedHandler { [weak self] commandBuffer in
                if let strongSelf = self {
                    strongSelf.inFlightSemaphore.signal()
                }
            }
            
            //meshAnchorも更新
            updateAppState()
            
            applyGaussianBlur(commandBuffer: commandBuffer)
            
            if meshFlag == true {
                //kernelVertex()
                update_kernelVertex()
            }
            
            
            // Pass the depth and confidence pixel buffers to the GPU to shade-in the fog.
            if let renderPassDescriptor = renderDestination.currentRenderPassDescriptor,
               let currentDrawable = renderDestination.currentDrawable {
                
                if let fogRenderEncoding = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                    
                    // Set a label to identify this render pass in a captured Metal frame.
                    fogRenderEncoding.label = "MyFogRenderEncoder"
                    
                    // Schedule the camera image and fog to be drawn to the screen.
                    doFogRenderPass(renderEncoder: fogRenderEncoding)
                    
                    if meshFlag == true {
                        drawMesh(renderEncoder: fogRenderEncoding)
                        
                        //calcuVertex(renderEncoder: fogRenderEncoding)
                    }
                    
                    // Finish encoding commands.
                    fogRenderEncoding.endEncoding()
                }
                
                // Schedule a present once the framebuffer is complete using the current drawable.
                commandBuffer.present(currentDrawable)
            }
            
            // Finalize rendering here & push the command buffer to the GPU.
            commandBuffer.commit()
        }
    }
    
    // MARK: - Private
    
    // Create and load our basic Metal state objects.
    func loadMetal() {
        // Set the default formats needed to render.
        renderDestination.colorPixelFormat = .bgra8Unorm
        renderDestination.sampleCount = 1
        
        // Create a vertex buffer with our image plane vertex data.
        let imagePlaneVertexDataCount = kImagePlaneVertexData.count * MemoryLayout<Float>.size
        imagePlaneVertexBuffer = device.makeBuffer(bytes: kImagePlaneVertexData, length: imagePlaneVertexDataCount, options: [])
        imagePlaneVertexBuffer.label = "ImagePlaneVertexBuffer"
        
        // Load all the shader files with a metal file extension in the project.
        let defaultLibrary = device.makeDefaultLibrary()!
        
        // Create a vertex descriptor for our image plane vertex buffer.
        let imagePlaneVertexDescriptor = MTLVertexDescriptor()
        
        // Positions.
        imagePlaneVertexDescriptor.attributes[0].format = .float2
        imagePlaneVertexDescriptor.attributes[0].offset = 0
        imagePlaneVertexDescriptor.attributes[0].bufferIndex = Int(kBufferIndexMeshPositions.rawValue)
        
        // Texture coordinates.
        imagePlaneVertexDescriptor.attributes[1].format = .float2
        imagePlaneVertexDescriptor.attributes[1].offset = 8
        imagePlaneVertexDescriptor.attributes[1].bufferIndex = Int(kBufferIndexMeshPositions.rawValue)
        
        // Buffer Layout.
        imagePlaneVertexDescriptor.layouts[0].stride = 16
        imagePlaneVertexDescriptor.layouts[0].stepRate = 1
        imagePlaneVertexDescriptor.layouts[0].stepFunction = .perVertex
        
        // Create camera image texture cache.
        var textureCache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
        cameraImageTextureCache = textureCache
        
        // Define the shaders that will render the camera image and fog on the GPU.
        let fogVertexFunction = defaultLibrary.makeFunction(name: "fogVertexTransform")!
        let fogFragmentFunction = defaultLibrary.makeFunction(name: "fogFragmentShader")!
        let fogPipelineStateDescriptor = MTLRenderPipelineDescriptor()
        fogPipelineStateDescriptor.label = "MyFogPipeline"
        fogPipelineStateDescriptor.sampleCount = renderDestination.sampleCount
        fogPipelineStateDescriptor.vertexFunction = fogVertexFunction
        fogPipelineStateDescriptor.fragmentFunction = fogFragmentFunction
        fogPipelineStateDescriptor.vertexDescriptor = imagePlaneVertexDescriptor
        fogPipelineStateDescriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        
        // Initialize the pipeline.
        do {
            try fogPipelineState = device.makeRenderPipelineState(descriptor: fogPipelineStateDescriptor)
        } catch let error {
            print("Failed to create fog pipeline state, error \(error)")
        }
        
        let meshVertexFunction = defaultLibrary.makeFunction(name: "MeshVertex")!
        let meshFragmentFunction = defaultLibrary.makeFunction(name: "MeshFragment")!
        let meshPipelineStateDescriptor = MTLRenderPipelineDescriptor()
        meshPipelineStateDescriptor.label = "MymeshPipeline"
        meshPipelineStateDescriptor.sampleCount = renderDestination.sampleCount
        meshPipelineStateDescriptor.vertexFunction = meshVertexFunction
        meshPipelineStateDescriptor.fragmentFunction = meshFragmentFunction
        meshPipelineStateDescriptor.vertexDescriptor = imagePlaneVertexDescriptor
        meshPipelineStateDescriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        do {
            try meshPipelineState = device.makeRenderPipelineState(descriptor: meshPipelineStateDescriptor)
        } catch let error {
            print("Failed to create mesh pipeline state, error \(error)")
        }
        
        let calcuVertexFunction = defaultLibrary.makeFunction(name: "CalcuVertex")!
        let calcuPipelineStateDescriptor = MTLRenderPipelineDescriptor()
        calcuPipelineStateDescriptor.label = "MycalcuPipeline"
        //calcuPipelineStateDescriptor.sampleCount = renderDestination.sampleCount
        calcuPipelineStateDescriptor.vertexFunction = calcuVertexFunction
        calcuPipelineStateDescriptor.isRasterizationEnabled = false
        //calcuPipelineStateDescriptor.vertexDescriptor = imagePlaneVertexDescriptor
        calcuPipelineStateDescriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        do {
            try calcuPipelineState = device.makeRenderPipelineState(descriptor: calcuPipelineStateDescriptor)
        } catch let error {
            print("Failed to create calcu pipeline state, error \(error)")
        }
        
        // Create the command queue for one frame of rendering work.
        commandQueue = device.makeCommandQueue()
        
        //kernel設定
        let function = defaultLibrary.makeFunction(name: "KernelVertex")!
        kernelPipeline = try! device.makeComputePipelineState(function: function)
        
        let function2 = defaultLibrary.makeFunction(name: "UpdateKernelVertex")!
        update_kernelPipeline = try! device.makeComputePipelineState(function: function2)
    }
    
    // Updates any app state.
    func updateAppState() {
        
        // Get the AR session's current frame.
        guard let currentFrame = session.currentFrame else {
            return
        }
        
        //ARMeshAnchorを格納
        let meshAnchors = currentFrame.anchors.compactMap { $0 as? ARMeshAnchor }
        //print(meshAnchors)
        if meshAnchors.count > 0 {
            for meshanchor in meshAnchors {
                if meshanchor.identifier == ID {
                    if anchorID[ID] != nil {
                        anchor = meshanchor
                        anchorNum = anchorID[ID]!
                        meshFlag = true
                    } else {
                        meshFlag = false
                    }
                }
            }
        }
        
        let orientation = UIInterfaceOrientation.landscapeRight //portrait
        let viewMatrix = currentFrame.camera.viewMatrix(for: orientation)
        let projectionMatrix = currentFrame.camera.projectionMatrix(for: orientation, viewportSize: viewportSize, zNear: 0.001, zFar: 0)
        viewProjectionMatrix = projectionMatrix * viewMatrix
        
        // Prepare the current frame's camera image for transfer to the GPU.
        updateCameraImageTextures(frame: currentFrame)
        
        // Prepare the current frame's depth and confidence images for transfer to the GPU.
        updateARDepthTexures(frame: currentFrame)
        
        // Update the destination-rendering vertex info if the size of the screen changed.
        if viewportSizeDidChange {
            viewportSizeDidChange = false
            updateImagePlane(frame: currentFrame)
        }
    }
    
    // Creates two textures (Y and CbCr) to transfer the current frame's camera image to the GPU for rendering.
    func updateCameraImageTextures(frame: ARFrame) {
        if CVPixelBufferGetPlaneCount(frame.capturedImage) < 2 {
            return
        }
        cameraImageTextureY = createTexture(fromPixelBuffer: frame.capturedImage, pixelFormat: .r8Unorm, planeIndex: 0)
        cameraImageTextureCbCr = createTexture(fromPixelBuffer: frame.capturedImage, pixelFormat: .rg8Unorm, planeIndex: 1)
    }
    
    // Assigns an appropriate MTL pixel format given the argument pixel-buffer's format.
    fileprivate func setMTLPixelFormat(_ texturePixelFormat: inout MTLPixelFormat?, basedOn pixelBuffer: CVPixelBuffer!) {
        if CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_DepthFloat32 {
            texturePixelFormat = .r32Float
        } else if CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_OneComponent8 {
            texturePixelFormat = .r8Uint
        } else {
            fatalError("Unsupported ARDepthData pixel-buffer format.")
        }
    }
    
    // Prepares the scene depth information for transfer to the GPU for rendering.
    func updateARDepthTexures(frame: ARFrame) {
        // Get the scene depth or smoothed scene depth from the current frame.
        guard let sceneDepth = frame.smoothedSceneDepth ?? frame.sceneDepth else {
            print("Failed to acquire scene depth.")
            return
        }
        var pixelBuffer: CVPixelBuffer!
        pixelBuffer = sceneDepth.depthMap
        
        // Set up the destination pixel format for the depth information, and
        // create a Metal texture from the depth image provided by ARKit.
        var texturePixelFormat: MTLPixelFormat!
        setMTLPixelFormat(&texturePixelFormat, basedOn: pixelBuffer)
        depthTexture = createTexture(fromPixelBuffer: pixelBuffer, pixelFormat: texturePixelFormat, planeIndex: 0)
        
        // Get the current depth confidence values from the current frame.
        // Set up the destination pixel format for the confidence information, and
        // create a Metal texture from the confidence image provided by ARKit.
        pixelBuffer = sceneDepth.confidenceMap
        setMTLPixelFormat(&texturePixelFormat, basedOn: pixelBuffer)
        confidenceTexture = createTexture(fromPixelBuffer: pixelBuffer, pixelFormat: texturePixelFormat, planeIndex: 0)
    }
    
    // Creates a Metal texture with the argument pixel format from a CVPixelBuffer at the argument plane index.
    func createTexture(fromPixelBuffer pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat, planeIndex: Int) -> CVMetalTexture? {
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)
        
        var texture: CVMetalTexture? = nil
        let status = CVMetalTextureCacheCreateTextureFromImage(nil, cameraImageTextureCache, pixelBuffer, nil, pixelFormat,
                                                               width, height, planeIndex, &texture)
        
        if status != kCVReturnSuccess {
            texture = nil
        }
        
        return texture
    }
    
    // Sets up vertex data (source and destination rectangles) rendering.
    func updateImagePlane(frame: ARFrame) {
        // Update the texture coordinates of the image plane to aspect fill the viewport.
        let displayToCameraTransform = frame.displayTransform(for: .landscapeRight, viewportSize: viewportSize).inverted()
        let vertexData = imagePlaneVertexBuffer.contents().assumingMemoryBound(to: Float.self)
        let fogVertexData = imagePlaneVertexBuffer.contents().assumingMemoryBound(to: Float.self)
        for index in 0...3 {
            let textureCoordIndex = 4 * index + 2
            let textureCoord = CGPoint(x: CGFloat(kImagePlaneVertexData[textureCoordIndex]), y: CGFloat(kImagePlaneVertexData[textureCoordIndex + 1]))
            let transformedCoord = textureCoord.applying(displayToCameraTransform)
            vertexData[textureCoordIndex] = Float(transformedCoord.x)
            vertexData[textureCoordIndex + 1] = Float(transformedCoord.y)
            fogVertexData[textureCoordIndex] = Float(transformedCoord.x)
            fogVertexData[textureCoordIndex + 1] = Float(transformedCoord.y)
        }
    }
    
    // Schedules the camera image and fog to be rendered on the GPU.
    func doFogRenderPass(renderEncoder: MTLRenderCommandEncoder) {
        guard let cameraImageY = cameraImageTextureY, let cameraImageCbCr = cameraImageTextureCbCr,
              let confidenceTexture = confidenceTexture else {
                  return
              }
        
        // Push a debug group that enables you to identify this render pass in a Metal frame capture.
        renderEncoder.pushDebugGroup("FogPass")
        
        // Set render command encoder state.
        renderEncoder.setCullMode(.none)
        renderEncoder.setRenderPipelineState(fogPipelineState)
        
        // Setup plane vertex buffers.
        renderEncoder.setVertexBuffer(imagePlaneVertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(imagePlaneVertexBuffer, offset: 0, index: 1)
        
        // Setup textures for the fog fragment shader.
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(cameraImageY), index: 0)
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(cameraImageCbCr), index: 1)
        renderEncoder.setFragmentTexture(filteredDepthTexture, index: 2)
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(confidenceTexture), index: 3)
        // Draw final quad to display
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.popDebugGroup()
    }
    
    func drawMesh(renderEncoder: MTLRenderCommandEncoder) {
        guard let cameraImageY = cameraImageTextureY,
              let cameraImageCbCr = cameraImageTextureCbCr
        else {
            return
        }
        
        renderEncoder.pushDebugGroup("MeshPass")
        
        // Set render command encoder state.
        renderEncoder.setCullMode(.none) //向き付け
        renderEncoder.setRenderPipelineState(meshPipelineState)
        
        renderEncoder.setVertexBuffer(anchor.geometry.vertices.buffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(anchor.geometry.faces.buffer, offset: 0, index: 1)
        
        let entity: AnchorUniforms = AnchorUniforms(transform: anchor.transform, viewProjectionMatrix: viewProjectionMatrix)
        anchorUniforms = entity
        anchorUniformsBuffer = device.makeBuffer(bytes: [anchorUniforms], length: MemoryLayout<AnchorUniforms>.size, options: [])
        renderEncoder.setVertexBuffer(anchorUniformsBuffer, offset: 0, index: 2)
        
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(cameraImageY), index: 3)
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(cameraImageCbCr), index: 4)
        
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: anchor.geometry.faces.count * 3)
        renderEncoder.popDebugGroup()
    }
    
    func calcuVertex(renderEncoder: MTLRenderCommandEncoder) {
        
        renderEncoder.pushDebugGroup("CalcuPass")
        
        renderEncoder.setCullMode(.none)
        renderEncoder.setDepthStencilState(relaxedStencilState)
        renderEncoder.setRenderPipelineState(calcuPipelineState)
        
        renderEncoder.setVertexBuffer(anchor.geometry.vertices.buffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(anchor.geometry.faces.buffer, offset: 0, index: 1)
        
        pre_vertexUniformsBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride * anchor.geometry.faces.count * 3, options: []) //.init(device: device, count: anchor.geometry.faces.count * 3, index: 2)
        renderEncoder.setVertexBuffer(pre_vertexUniformsBuffer, offset: 0, index: 2) //index = 2
        
        //        let tryBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride * anchor.geometry.faces.count * 3, options: [])
        //        renderEncoder.setVertexBuffer(tryBuffer, offset: 0, index: 6)
        
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: anchor.geometry.faces.count * 3)
        renderEncoder.popDebugGroup()
        
        
        //print(pre_vertexUniformsBuffer[0])
    }
    
    func kernelVertex() { //renderEncoder: MTLComputeCommandEncoder
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let renderEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        renderEncoder.pushDebugGroup("CalcuPass")
        
        renderEncoder.setComputePipelineState(kernelPipeline)
        
        renderEncoder.setBuffer(anchor.geometry.vertices.buffer, offset: 0, index: 0)
        renderEncoder.setBuffer(anchor.geometry.faces.buffer, offset: 0, index: 1)
        
        pre_vertexUniformsBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride * anchor.geometry.faces.count * 3, options: []) //.init(device: device, count: anchor.geometry.faces.count * 3, index: 2)
        renderEncoder.setBuffer(pre_vertexUniformsBuffer, offset: 0, index: 2) //index = 2
        
        renderEncoder.setBuffer(vertexUniformsBuffer, offset: 0, index: 3)
        
        
        sumCount += anchor.geometry.faces.count * 3
        
        let num = anchorID[ID]
        let entity: CalcuUniforms = CalcuUniforms(pre_count: Int32(perFaceCount[num!]),
                                                  new_count: Int32(anchor.geometry.faces.count * 3),
                                                  left_sum: Int32(perFaceIndex[num!]))
        renderEncoder.setBuffer(device.makeBuffer(bytes: [entity], length: MemoryLayout<CalcuUniforms>.size, options: []), offset: 0, index: 4)
        
        //        let tryBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride * anchor.geometry.faces.count * 3, options: [])
        //        renderEncoder.setVertexBuffer(tryBuffer, offset: 0, index: 6)
        
        let width = 1//32
        let threadsPerGroup = MTLSize(width: width, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (anchor.geometry.faces.count + width - 1) / width, height: 1, depth: 1)
        renderEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        renderEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        renderEncoder.popDebugGroup()
        
//        let tryData = Data(bytesNoCopy: pre_vertexUniformsBuffer!.contents(), count: MemoryLayout<SIMD3<Float>>.stride * anchor.geometry.faces.count * 3, deallocator: .none)
//        var trys = [SIMD3<Float>](repeating: SIMD3<Float>(0,0,0), count: anchor.geometry.faces.count * 3)
//        trys = tryData.withUnsafeBytes {
//            Array(UnsafeBufferPointer<SIMD3<Float>>(start: $0, count: tryData.count/MemoryLayout<SIMD3<Float>>.size))
//        }
//        print(trys)
        //print(pre_vertexUniformsBuffer[0])
    }
    
    func update_kernelVertex() {
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let renderEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        renderEncoder.pushDebugGroup("CalcuPass2")
        
        renderEncoder.setComputePipelineState(update_kernelPipeline)
        
        renderEncoder.setBuffer(anchor.geometry.vertices.buffer, offset: 0, index: 0)
        renderEncoder.setBuffer(anchor.geometry.faces.buffer, offset: 0, index: 1)
        
        pre_vertexUniformsBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride * anchor.geometry.faces.count * 3, options: []) //.init(device: device, count: anchor.geometry.faces.count * 3, index: 2)
        renderEncoder.setBuffer(pre_vertexUniformsBuffer, offset: 0, index: 2) //index = 2
        
        renderEncoder.setBuffer(vertexUniformsBuffer, offset: 0, index: 3)
        
        let num = anchorNum
        print("num : \(num)")
        let entity: CalcuUniforms = CalcuUniforms(pre_count: Int32(perFaceCount[num!]),
                                                  new_count: Int32(anchor.geometry.faces.count * 3),
                                                  left_sum: Int32(perFaceIndex[num!]))
        renderEncoder.setBuffer(device.makeBuffer(bytes: [entity], length: MemoryLayout<CalcuUniforms>.size, options: []), offset: 0, index: 4)
        
        //sumCount += anchor.geometry.faces.count * 3 - perFaceCount[num!]
        for i in num!+1...anchorID.count {
            if i < perFaceIndex.count {
                perFaceIndex[i] += (anchor.geometry.faces.count * 3 - perFaceCount[num!])
            }
        }
        perFaceCount[num!] = anchor.geometry.faces.count * 3
        sumCount = perFaceIndex.last!
         
        print("anchorID : \(anchorID)")
        print("perFaceCount : \(perFaceCount)")
        print("perFaceIndex : \(perFaceIndex)")
        print("sumCount : \(sumCount)")
         
        //        let tryBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride * anchor.geometry.faces.count * 3, options: [])
        //        renderEncoder.setVertexBuffer(tryBuffer, offset: 0, index: 6)
        
        let width = 1//32
        let threadsPerGroup = MTLSize(width: width, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (sumCount + width - 1) / width, height: 1, depth: 1)
        renderEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        renderEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        renderEncoder.popDebugGroup()
    }
    
    // MARK: - MPS Filter
    
    // Sets up a filter to process the depth texture.
    func setupFilter(width: Int, height: Int) {
        // Create a destination backing-store to hold the blurred result.
        let filteredDepthDescriptor = MTLTextureDescriptor()
        filteredDepthDescriptor.pixelFormat = .r32Float
        filteredDepthDescriptor.width = width
        filteredDepthDescriptor.height = height
        filteredDepthDescriptor.usage = [.shaderRead, .shaderWrite]
        filteredDepthTexture = device.makeTexture(descriptor: filteredDepthDescriptor)
        blurFilter = MPSImageGaussianBlur(device: device, sigma: 5)
    }
    
    // Schedules the depth texture to be blurred on the GPU using the `blurFilter`.
    func applyGaussianBlur(commandBuffer: MTLCommandBuffer) {
        guard let arDepthTexture = depthTexture,
              let depthTexture = CVMetalTextureGetTexture(arDepthTexture)
        else {
            print("Error: Unable to apply the MPS filter.")
            return
        }
        //print("Able to apply the MPS filter.")
        guard let blur = blurFilter else {
            setupFilter(width: depthTexture.width, height: depthTexture.height)
            return
        }
        
        let inputImage = MPSImage(texture: depthTexture, featureChannels: 1)
        let outputImage = MPSImage(texture: filteredDepthTexture, featureChannels: 1)
        blur.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
    }
}
