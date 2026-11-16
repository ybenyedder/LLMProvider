//! Exact C-ABI JNI definitions matching Android NDK `jni.h`.
#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_void};

pub type jint = i32;
pub type jlong = i64;
pub type jbyte = i8;
pub type jboolean = u8;
pub type jchar = u16;
pub type jshort = i16;
pub type jfloat = f32;
pub type jdouble = f64;
pub type jsize = jint;

pub const JNI_FALSE: jboolean = 0;
pub const JNI_TRUE: jboolean = 1;
pub const JNI_OK: jint = 0;
pub const JNI_ERR: jint = -1;
pub const JNI_VERSION_1_6: jint = 0x00010006;

pub type jobject = *mut c_void;
pub type jclass = jobject;
pub type jthrowable = jobject;
pub type jstring = jobject;
pub type jarray = jobject;
pub type jbooleanArray = jarray;
pub type jbyteArray = jarray;
pub type jcharArray = jarray;
pub type jshortArray = jarray;
pub type jintArray = jarray;
pub type jlongArray = jarray;
pub type jfloatArray = jarray;
pub type jdoubleArray = jarray;
pub type jobjectArray = jarray;
pub type jweak = jobject;

pub type jfieldID = *mut c_void;
pub type jmethodID = *mut c_void;

#[repr(C)]
#[derive(Copy, Clone)]
pub union jvalue {
    pub z: jboolean,
    pub b: jbyte,
    pub c: jchar,
    pub s: jshort,
    pub i: jint,
    pub j: jlong,
    pub f: jfloat,
    pub d: jdouble,
    pub l: jobject,
}

pub type JNIEnv = *const JNINativeInterface;
pub type JavaVM = *const JNIInvokeInterface;

#[repr(C)]
pub struct JNIInvokeInterface {
    pub reserved0: *mut c_void,
    pub reserved1: *mut c_void,
    pub reserved2: *mut c_void,
    pub DestroyJavaVM: unsafe extern "C" fn(*mut JavaVM) -> jint,
    pub AttachCurrentThread: unsafe extern "C" fn(*mut JavaVM, *mut *mut JNIEnv, *mut c_void) -> jint,
    pub DetachCurrentThread: unsafe extern "C" fn(*mut JavaVM) -> jint,
    pub GetEnv: unsafe extern "C" fn(*mut JavaVM, *mut *mut c_void, jint) -> jint,
    pub AttachCurrentThreadAsDaemon:
        unsafe extern "C" fn(*mut JavaVM, *mut *mut JNIEnv, *mut c_void) -> jint,
}

type FnPtr = *const c_void;

#[repr(C)]
pub struct JNINativeInterface {
    pub reserved0: *mut c_void,
    pub reserved1: *mut c_void,
    pub reserved2: *mut c_void,
    pub reserved3: *mut c_void,

    pub GetVersion: unsafe extern "C" fn(*mut JNIEnv) -> jint,
    pub DefineClass: FnPtr,
    pub FindClass: unsafe extern "C" fn(*mut JNIEnv, *const c_char) -> jclass,
    pub FromReflectedMethod: FnPtr,
    pub FromReflectedField: FnPtr,
    pub ToReflectedMethod: FnPtr,
    pub GetSuperclass: FnPtr,
    pub IsAssignableFrom: FnPtr,
    pub ToReflectedField: FnPtr,

    pub Throw: FnPtr,
    pub ThrowNew: unsafe extern "C" fn(*mut JNIEnv, jclass, *const c_char) -> jint,
    pub ExceptionOccurred: FnPtr,
    pub ExceptionDescribe: unsafe extern "C" fn(*mut JNIEnv),
    pub ExceptionClear: unsafe extern "C" fn(*mut JNIEnv),
    pub FatalError: FnPtr,

    pub PushLocalFrame: FnPtr,
    pub PopLocalFrame: FnPtr,

    pub NewGlobalRef: unsafe extern "C" fn(*mut JNIEnv, jobject) -> jobject,
    pub DeleteGlobalRef: unsafe extern "C" fn(*mut JNIEnv, jobject),
    pub DeleteLocalRef: unsafe extern "C" fn(*mut JNIEnv, jobject),
    pub IsSameObject: FnPtr,
    pub NewLocalRef: FnPtr,
    pub EnsureLocalCapacity: FnPtr,

    pub AllocObject: FnPtr,
    pub NewObject: FnPtr,
    pub NewObjectV: FnPtr,
    pub NewObjectA: FnPtr,

    pub GetObjectClass: unsafe extern "C" fn(*mut JNIEnv, jobject) -> jclass,
    pub IsInstanceOf: FnPtr,
    pub GetMethodID:
        unsafe extern "C" fn(*mut JNIEnv, jclass, *const c_char, *const c_char) -> jmethodID,

    // 30 Call*Method slots (indices 34..63)
    pub _call_methods: [FnPtr; 29],
    pub CallVoidMethodA:
        unsafe extern "C" fn(*mut JNIEnv, jobject, jmethodID, *const jvalue),

    // 30 CallNonvirtual*Method slots (indices 64..93)
    pub _call_nonvirtual_methods: [FnPtr; 30],

    pub GetFieldID: FnPtr,
    pub _get_fields: [FnPtr; 9],
    pub _set_fields: [FnPtr; 9],
    pub GetStaticMethodID: FnPtr,
    pub _call_static_methods: [FnPtr; 30],
    pub GetStaticFieldID: FnPtr,
    pub _get_static_fields: [FnPtr; 9],
    pub _set_static_fields: [FnPtr; 9],

    // String & Array functions (index 163..)
    pub NewString: unsafe extern "C" fn(*mut JNIEnv, *const jchar, jsize) -> jstring,
    pub GetStringLength: unsafe extern "C" fn(*mut JNIEnv, jstring) -> jsize,
    pub GetStringChars: unsafe extern "C" fn(*mut JNIEnv, jstring, *mut jboolean) -> *const jchar,
    pub ReleaseStringChars: unsafe extern "C" fn(*mut JNIEnv, jstring, *const jchar),
    pub NewStringUTF: unsafe extern "C" fn(*mut JNIEnv, *const c_char) -> jstring,
    pub GetStringUTFLength: unsafe extern "C" fn(*mut JNIEnv, jstring) -> jsize,
    pub GetStringUTFChars:
        unsafe extern "C" fn(*mut JNIEnv, jstring, *mut jboolean) -> *const c_char,
    pub ReleaseStringUTFChars: unsafe extern "C" fn(*mut JNIEnv, jstring, *const c_char),

    pub GetArrayLength: unsafe extern "C" fn(*mut JNIEnv, jarray) -> jsize,
    pub NewObjectArray: FnPtr,
    pub GetObjectArrayElement: FnPtr,
    pub SetObjectArrayElement: FnPtr,

    pub NewBooleanArray: FnPtr,
    pub NewByteArray: FnPtr,
    pub NewCharArray: FnPtr,
    pub NewShortArray: FnPtr,
    pub NewIntArray: unsafe extern "C" fn(*mut JNIEnv, jsize) -> jintArray,
    pub NewLongArray: FnPtr,
    pub NewFloatArray: unsafe extern "C" fn(*mut JNIEnv, jsize) -> jfloatArray,
    pub NewDoubleArray: FnPtr,

    pub GetBooleanArrayElements: FnPtr,
    pub GetByteArrayElements: FnPtr,
    pub GetCharArrayElements: FnPtr,
    pub GetShortArrayElements: FnPtr,
    pub GetIntArrayElements: FnPtr,
    pub GetLongArrayElements: FnPtr,
    pub GetFloatArrayElements: FnPtr,
    pub GetDoubleArrayElements: FnPtr,

    pub ReleaseBooleanArrayElements: FnPtr,
    pub ReleaseByteArrayElements: FnPtr,
    pub ReleaseCharArrayElements: FnPtr,
    pub ReleaseShortArrayElements: FnPtr,
    pub ReleaseIntArrayElements: FnPtr,
    pub ReleaseLongArrayElements: FnPtr,
    pub ReleaseFloatArrayElements: FnPtr,
    pub ReleaseDoubleArrayElements: FnPtr,

    pub GetBooleanArrayRegion: FnPtr,
    pub GetByteArrayRegion: FnPtr,
    pub GetCharArrayRegion: FnPtr,
    pub GetShortArrayRegion: FnPtr,
    pub GetIntArrayRegion:
        unsafe extern "C" fn(*mut JNIEnv, jintArray, jsize, jsize, *mut jint),
    pub GetLongArrayRegion: FnPtr,
    pub GetFloatArrayRegion:
        unsafe extern "C" fn(*mut JNIEnv, jfloatArray, jsize, jsize, *mut jfloat),
    pub GetDoubleArrayRegion: FnPtr,

    pub SetBooleanArrayRegion: FnPtr,
    pub SetByteArrayRegion: FnPtr,
    pub SetCharArrayRegion: FnPtr,
    pub SetShortArrayRegion: FnPtr,
    pub SetIntArrayRegion:
        unsafe extern "C" fn(*mut JNIEnv, jintArray, jsize, jsize, *const jint),
    pub SetLongArrayRegion: FnPtr,
    pub SetFloatArrayRegion:
        unsafe extern "C" fn(*mut JNIEnv, jfloatArray, jsize, jsize, *const jfloat),
    pub SetDoubleArrayRegion: FnPtr,

    pub RegisterNatives: FnPtr,
    pub UnregisterNatives: FnPtr,
    pub MonitorEnter: FnPtr,
    pub MonitorExit: FnPtr,
    pub GetJavaVM: FnPtr,

    pub GetStringRegion: FnPtr,
    pub GetStringUTFRegion: FnPtr,
    pub GetPrimitiveArrayCritical: FnPtr,
    pub ReleasePrimitiveArrayCritical: FnPtr,
    pub GetStringCritical: FnPtr,
    pub ReleaseStringCritical: FnPtr,
    pub NewWeakGlobalRef: FnPtr,
    pub DeleteWeakGlobalRef: FnPtr,
    pub ExceptionCheck: unsafe extern "C" fn(*mut JNIEnv) -> jboolean,
    pub NewDirectByteBuffer: FnPtr,
    pub GetDirectBufferAddress: FnPtr,
    pub GetDirectBufferCapacity: FnPtr,
    pub GetObjectRefType: FnPtr,
}
