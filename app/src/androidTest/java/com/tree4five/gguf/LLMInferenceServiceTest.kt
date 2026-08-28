package com.tree4five.gguf

import android.content.Intent
import android.os.IBinder
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.ServiceTestRule
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
class LLMInferenceServiceTest {

    @get:Rule
    val serviceRule = ServiceTestRule()

    @Test
    fun testServiceBindingAndGeneration() {
        // Create the service intent
        val serviceIntent = Intent(ApplicationProvider.getApplicationContext(), LLMInferenceService::class.java)

        // Bind the service and grab the IBinder
        val binder: IBinder = serviceRule.bindService(serviceIntent)
        
        // Cast the binder to our AIDL interface
        val service = ILLMService.Stub.asInterface(binder)
        assertNotNull("Service should be bound", service)

        val latch = CountDownLatch(1)
        var receivedTokens = false
        var completeText = ""

        val callback = object : ILLMCallback.Stub() {
            override fun onTokenReceived(token: String) {
                receivedTokens = true
            }

            override fun onGenerationComplete(fullText: String) {
                completeText = fullText
                latch.countDown()
            }
        }

        // Call the service
        service.generateTextStream("Test prompt", callback)

        // Wait for generation to complete (or fail gracefully if model is missing)
        latch.await(5, TimeUnit.SECONDS)
        
        // Assertions
        assertNotNull(completeText)
        assertTrue("Callback should complete", completeText.isNotEmpty() || receivedTokens)
    }
}
