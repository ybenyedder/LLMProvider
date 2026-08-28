package com.example.gguf

import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class ModelDownloaderTest {

    @Test
    fun testDownloadFailsGracefullyOnBadUrl() {
        runBlocking {
            val badUrl = "https://nonexistent.domain/fake.gguf"
            val tempFile = File.createTempFile("test_model", ".gguf")
            tempFile.deleteOnExit()

            val success = ModelDownloader.downloadOrCopyModel(badUrl, tempFile, null) { progress ->
                // Do nothing
            }

            assertFalse("Download should fail for a bad URL", success)
            assertFalse("File should be deleted or not exist on failure", tempFile.exists() && tempFile.length() > 0)
        }
    }
}
