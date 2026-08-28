/*
 * This file is auto-generated.  DO NOT MODIFY.
 */
package com.example.gguf;
public interface ILLMService extends android.os.IInterface
{
  /** Default implementation for ILLMService. */
  public static class Default implements com.example.gguf.ILLMService
  {
    /** Starts text generation and streams the output to the provided callback. */
    @Override public void generateTextStream(java.lang.String prompt, com.example.gguf.ILLMCallback callback) throws android.os.RemoteException
    {
    }
    @Override
    public android.os.IBinder asBinder() {
      return null;
    }
  }
  /** Local-side IPC implementation stub class. */
  public static abstract class Stub extends android.os.Binder implements com.example.gguf.ILLMService
  {
    /** Construct the stub at attach it to the interface. */
    public Stub()
    {
      this.attachInterface(this, DESCRIPTOR);
    }
    /**
     * Cast an IBinder object into an com.example.gguf.ILLMService interface,
     * generating a proxy if needed.
     */
    public static com.example.gguf.ILLMService asInterface(android.os.IBinder obj)
    {
      if ((obj==null)) {
        return null;
      }
      android.os.IInterface iin = obj.queryLocalInterface(DESCRIPTOR);
      if (((iin!=null)&&(iin instanceof com.example.gguf.ILLMService))) {
        return ((com.example.gguf.ILLMService)iin);
      }
      return new com.example.gguf.ILLMService.Stub.Proxy(obj);
    }
    @Override public android.os.IBinder asBinder()
    {
      return this;
    }
    @Override public boolean onTransact(int code, android.os.Parcel data, android.os.Parcel reply, int flags) throws android.os.RemoteException
    {
      java.lang.String descriptor = DESCRIPTOR;
      if (code >= android.os.IBinder.FIRST_CALL_TRANSACTION && code <= android.os.IBinder.LAST_CALL_TRANSACTION) {
        data.enforceInterface(descriptor);
      }
      switch (code)
      {
        case INTERFACE_TRANSACTION:
        {
          reply.writeString(descriptor);
          return true;
        }
      }
      switch (code)
      {
        case TRANSACTION_generateTextStream:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          com.example.gguf.ILLMCallback _arg1;
          _arg1 = com.example.gguf.ILLMCallback.Stub.asInterface(data.readStrongBinder());
          this.generateTextStream(_arg0, _arg1);
          break;
        }
        default:
        {
          return super.onTransact(code, data, reply, flags);
        }
      }
      return true;
    }
    private static class Proxy implements com.example.gguf.ILLMService
    {
      private android.os.IBinder mRemote;
      Proxy(android.os.IBinder remote)
      {
        mRemote = remote;
      }
      @Override public android.os.IBinder asBinder()
      {
        return mRemote;
      }
      public java.lang.String getInterfaceDescriptor()
      {
        return DESCRIPTOR;
      }
      /** Starts text generation and streams the output to the provided callback. */
      @Override public void generateTextStream(java.lang.String prompt, com.example.gguf.ILLMCallback callback) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(prompt);
          _data.writeStrongInterface(callback);
          boolean _status = mRemote.transact(Stub.TRANSACTION_generateTextStream, _data, null, android.os.IBinder.FLAG_ONEWAY);
        }
        finally {
          _data.recycle();
        }
      }
    }
    static final int TRANSACTION_generateTextStream = (android.os.IBinder.FIRST_CALL_TRANSACTION + 0);
  }
  public static final java.lang.String DESCRIPTOR = "com.example.gguf.ILLMService";
  /** Starts text generation and streams the output to the provided callback. */
  public void generateTextStream(java.lang.String prompt, com.example.gguf.ILLMCallback callback) throws android.os.RemoteException;
}
