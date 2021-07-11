//
//  ContentView.swift
//  coreml-benchmark
//
//  Created by Jaewon Choi on 2021/06/25.
//

import SwiftUI
import CoreML

struct ContentView: View {
    var body: some View {
        Text("Hello, world!")
            .padding()
            .onAppear {
                do {
                    let model = try Siren(configuration: MLModelConfiguration())
                    let hidden_size = 64
                    let batch_size = 16
                    
                    let m1 = try? MLMultiArray(shape: [batch_size, 2, hidden_size] as [NSNumber], dataType: .float)
                    let m2 = try? MLMultiArray(shape: [batch_size, hidden_size, hidden_size] as [NSNumber], dataType: .float)
                    let m3 = try? MLMultiArray(shape: [batch_size, hidden_size, 3] as [NSNumber], dataType: .float)
                    
                    let b1 = try? MLMultiArray(shape: [batch_size, 1, hidden_size] as [NSNumber], dataType: .float)
                    let b2 = try? MLMultiArray(shape: [batch_size, 1, hidden_size] as [NSNumber], dataType: .float)
                    let b3 = try? MLMultiArray(shape: [batch_size, 1, 3] as [NSNumber], dataType: .float)
                    
                    let start = DispatchTime.now()
                    
                    for index in 1...1000 {
                        guard (try? model.prediction(input: SirenInput(m1: m1!, b1: b1!, m2: m2!, b2: b2!, m3: m3!, b3: b3!))) != nil else {
                            fatalError("Unexpected runtime error.")
                        }
                    }
                    
                    let end = DispatchTime.now()
                    
                    let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds // <<<<< Difference in nano seconds (UInt64)
                    let timeInterval = Double(nanoTime) / 1_000_000_000 // Technically could overflow for long running tests

                    print("Time to evaluate: \(timeInterval) seconds")
                } catch {
                    fatalError("Model initialization failed.")
                }
            }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
